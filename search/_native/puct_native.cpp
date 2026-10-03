// PUCT 的 C++ 实现：与 unichess_kit/search/puct.py 逐位一致，经 ctypes 调用（调用期间释放 GIL）。
//
// 对齐的对象与方法（每一条都有测试：tests/test_puct_cpp.py）：
// - 棋规：python-chess 1.11 的合法着法**顺序**（argmax 平局取第一个，顺序不同树就不同）、
//   push 语义（易位权、原始 ep 格、半步钟）、is_repetition（逐步回退，遇不可逆着法停）、
//   终局判定顺序（将死 → 申和 → 逼和 / 子力不足 / 75 步 / 五次重复）。
// - 数值：numpy 2.x（NEP 50）的类型提升——float32 数组与 Python float 运算得 float32、
//   int32 + float32 得 float64；数组求和是 numpy 的 pairwise 求和（8 路展开、128 分块）。
//   编译必须关掉 FMA 合并（-ffp-contract=off），否则乘加会少一次舍入。
// - 残局表探测留在 Python：收集叶子时遇到子力 <= max_pieces 的局面就暂停（kp_collect 返回 -1），
//   Python 探测后 kp_probe_answer 回填，再次调用 kp_collect 从断点继续。
//
// 编码约定：着法 = from | to << 6 | promo << 12（promo 用 python-chess 的棋子号：2N 3B 4R 5Q）。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

typedef uint64_t BB;
inline BB bit(int s) { return 1ULL << s; }
inline int lsb(BB b) { return __builtin_ctzll(b); }
inline int msb(BB b) { return 63 - __builtin_clzll(b); }
inline int popcnt(BB b) { return __builtin_popcountll(b); }

enum { NONE = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };
enum { BLACK = 0, WHITE = 1 };

const BB RANK_1 = 0xFFULL;
const BB RANK_8 = 0xFFULL << 56;
const BB DARK_SQUARES = 0xAA55AA55AA55AA55ULL;
const BB LIGHT_SQUARES = 0x55AA55AA55AA55AAULL;
BB RANK_BB[8], FILE_BB[8];
BB KNIGHT_ATT[64], KING_ATT[64], PAWN_ATT[2][64];

BB step_mask(int sq, const int (*d)[2], int n) {
    BB m = 0;
    int r = sq >> 3, f = sq & 7;
    for (int i = 0; i < n; ++i) {
        int rr = r + d[i][0], ff = f + d[i][1];
        if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) m |= bit(rr * 8 + ff);
    }
    return m;
}

struct Tables {
    Tables() {
        static const int KN[8][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
        static const int KG[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        static const int PW[2][2] = {{1, -1}, {1, 1}};
        static const int PB[2][2] = {{-1, -1}, {-1, 1}};
        for (int i = 0; i < 8; ++i) {
            RANK_BB[i] = 0xFFULL << (8 * i);
            FILE_BB[i] = 0x0101010101010101ULL << i;
        }
        for (int s = 0; s < 64; ++s) {
            KNIGHT_ATT[s] = step_mask(s, KN, 8);
            KING_ATT[s] = step_mask(s, KG, 8);
            PAWN_ATT[WHITE][s] = step_mask(s, PW, 2);
            PAWN_ATT[BLACK][s] = step_mask(s, PB, 2);
        }
    }
} tables_;

const int ROOK_DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
const int BISHOP_DIRS[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

inline BB slide(int sq, BB occ, const int (*d)[2]) {
    BB a = 0;
    int r0 = sq >> 3, f0 = sq & 7;
    for (int i = 0; i < 4; ++i) {
        int r = r0 + d[i][0], f = f0 + d[i][1];
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            int s = r * 8 + f;
            a |= bit(s);
            if (occ & bit(s)) break;
            r += d[i][0];
            f += d[i][1];
        }
    }
    return a;
}
inline BB rook_att(int sq, BB occ) { return slide(sq, occ, ROOK_DIRS); }
inline BB bishop_att(int sq, BB occ) { return slide(sq, occ, BISHOP_DIRS); }

// python-chess between(a, b)：同线时严格位于两者之间的格，否则 0
BB between(int a, int b) {
    if (a == b) return 0;
    int ra = a >> 3, fa = a & 7, rb = b >> 3, fb = b & 7;
    int dr = (rb > ra) - (rb < ra), df = (fb > fa) - (fb < fa);
    if (!(ra == rb || fa == fb || std::abs(ra - rb) == std::abs(fa - fb))) return 0;
    BB m = 0;
    int r = ra + dr, f = fa + df;
    while (r != rb || f != fb) {
        m |= bit(r * 8 + f);
        r += dr;
        f += df;
    }
    return m;
}

struct Mv {
    int from, to, promo;
};
inline uint16_t mv_code(const Mv& m) { return (uint16_t)(m.from | (m.to << 6) | (m.promo << 12)); }
inline Mv mv_decode(uint16_t c) { return Mv{c & 63, (c >> 6) & 63, (c >> 12) & 7}; }

struct Pos {
    BB bb[7];     // 按棋子号 1..6（PAWN..KING）
    BB co[2];     // co[WHITE] / co[BLACK]
    int turn;
    BB castling;  // 恒为 python-chess 的 clean_castling_rights（载入时清洗，push 按原规则更新）
    int ep;       // 原始 ep 格（双步后总会设置，不论能否吃），-1 = 无
    int hmc;

    BB occ() const { return co[0] | co[1]; }
    int type_at(int sq) const {
        BB m = bit(sq);
        if (!(occ() & m)) return NONE;
        for (int t = PAWN; t <= KING; ++t)
            if (bb[t] & m) return t;
        return NONE;
    }
    void remove(int sq) {
        BB m = ~bit(sq);
        for (int t = PAWN; t <= KING; ++t) bb[t] &= m;
        co[0] &= m;
        co[1] &= m;
    }
    void put(int sq, int t, int color) {
        remove(sq);
        bb[t] |= bit(sq);
        co[color] |= bit(sq);
    }
};

BB attackers(const Pos& p, int color, int sq, BB occ) {
    BB a = (KING_ATT[sq] & p.bb[KING]) | (KNIGHT_ATT[sq] & p.bb[KNIGHT]) |
           (rook_att(sq, occ) & (p.bb[ROOK] | p.bb[QUEEN])) |
           (bishop_att(sq, occ) & (p.bb[BISHOP] | p.bb[QUEEN])) |
           (PAWN_ATT[color ^ 1][sq] & p.bb[PAWN]);
    return a & p.co[color];
}

bool is_check(const Pos& p) {
    BB k = p.bb[KING] & p.co[p.turn];
    return k && attackers(p, p.turn ^ 1, msb(k), p.occ());
}

BB clean_castling(const Pos& p, BB raw) {
    BB castling = raw & p.bb[ROOK];
    BB w = castling & RANK_1 & p.co[WHITE] & (bit(0) | bit(7));
    BB b = castling & RANK_8 & p.co[BLACK] & (bit(56) | bit(63));
    if (!(p.co[WHITE] & p.bb[KING] & bit(4))) w = 0;
    if (!(p.co[BLACK] & p.bb[KING] & bit(60))) b = 0;
    return w | b;
}

// ---------- 着法生成（python-chess 顺序）----------

void gen_castling(const Pos& p, std::vector<Mv>& out) {
    int us = p.turn;
    BB backrank = us == WHITE ? RANK_1 : RANK_8;
    BB king = p.co[us] & p.bb[KING] & backrank;
    king &= (~king + 1);
    if (!king) return;
    BB occ = p.occ();
    BB bb_c = FILE_BB[2] & backrank, bb_d = FILE_BB[3] & backrank;
    BB bb_f = FILE_BB[5] & backrank, bb_g = FILE_BB[6] & backrank;
    BB cands = p.castling & backrank;
    while (cands) {
        int cand = msb(cands);
        cands &= ~bit(cand);
        BB rook = bit(cand);
        bool a_side = rook < king;
        BB king_to = a_side ? bb_c : bb_g;
        BB rook_to = a_side ? bb_d : bb_f;
        BB king_path = between(msb(king), msb(king_to));
        BB rook_path = between(cand, msb(rook_to));
        bool blocked = ((occ ^ king ^ rook) & (king_path | rook_path | king_to | rook_to)) != 0;
        bool attacked = false;
        if (!blocked) {
            BB path = king_path | king, o1 = occ ^ king;
            while (path && !attacked) {
                int s = msb(path);
                path &= ~bit(s);
                if (attackers(p, us ^ 1, s, o1)) attacked = true;
            }
            BB kt = king_to, o2 = occ ^ king ^ rook ^ rook_to;
            while (kt && !attacked) {
                int s = msb(kt);
                kt &= ~bit(s);
                if (attackers(p, us ^ 1, s, o2)) attacked = true;
            }
        }
        if (!blocked && !attacked) {
            int ks = msb(king);
            int to = cand;  // _from_chess960：标准棋里 e1h1 → e1g1、e1a1 → e1c1
            if (ks == 4 && cand == 7) to = 6;
            else if (ks == 4 && cand == 0) to = 2;
            else if (ks == 60 && cand == 63) to = 62;
            else if (ks == 60 && cand == 56) to = 58;
            out.push_back(Mv{ks, to, 0});
        }
    }
}

void gen_ep(const Pos& p, std::vector<Mv>& out) {
    if (p.ep <= 0) return;  // python: `if not self.ep_square`
    if (bit(p.ep) & p.occ()) return;
    int us = p.turn;
    BB cap = p.bb[PAWN] & p.co[us] & PAWN_ATT[us ^ 1][p.ep] & RANK_BB[us == WHITE ? 4 : 3];
    while (cap) {
        int s = msb(cap);
        cap &= ~bit(s);
        out.push_back(Mv{s, p.ep, 0});
    }
}

inline void push_pawn(std::vector<Mv>& out, int from, int to) {
    int r = to >> 3;
    if (r == 0 || r == 7) {
        out.push_back(Mv{from, to, QUEEN});
        out.push_back(Mv{from, to, ROOK});
        out.push_back(Mv{from, to, BISHOP});
        out.push_back(Mv{from, to, KNIGHT});
    } else {
        out.push_back(Mv{from, to, 0});
    }
}

void gen_pseudo(const Pos& p, std::vector<Mv>& out) {
    int us = p.turn, them = us ^ 1;
    BB our = p.co[us], occ = p.occ();
    BB non_pawns = our & ~p.bb[PAWN];
    while (non_pawns) {
        int from = msb(non_pawns);
        non_pawns &= ~bit(from);
        int t = p.type_at(from);
        BB a = 0;
        if (t == KNIGHT) a = KNIGHT_ATT[from];
        else if (t == KING) a = KING_ATT[from];
        else if (t == ROOK) a = rook_att(from, occ);
        else if (t == BISHOP) a = bishop_att(from, occ);
        else if (t == QUEEN) a = rook_att(from, occ) | bishop_att(from, occ);
        a &= ~our;
        while (a) {
            int to = msb(a);
            a &= ~bit(to);
            out.push_back(Mv{from, to, 0});
        }
    }
    if (p.bb[KING]) gen_castling(p, out);
    BB pawns = p.bb[PAWN] & our;
    if (!pawns) return;
    BB cap = pawns;
    while (cap) {
        int from = msb(cap);
        cap &= ~bit(from);
        BB t = PAWN_ATT[us][from] & p.co[them];
        while (t) {
            int to = msb(t);
            t &= ~bit(to);
            push_pawn(out, from, to);
        }
    }
    BB single, dbl;
    if (us == WHITE) {
        single = (pawns << 8) & ~occ;
        dbl = (single << 8) & ~occ & (RANK_BB[2] | RANK_BB[3]);
    } else {
        single = (pawns >> 8) & ~occ;
        dbl = (single >> 8) & ~occ & (RANK_BB[5] | RANK_BB[4]);
    }
    while (single) {
        int to = msb(single);
        single &= ~bit(to);
        push_pawn(out, to + (us == BLACK ? 8 : -8), to);
    }
    while (dbl) {
        int to = msb(dbl);
        dbl &= ~bit(to);
        out.push_back(Mv{to + (us == BLACK ? 16 : -16), to, 0});
    }
    if (p.ep > 0) gen_ep(p, out);
}

inline bool is_castle_mv(const Pos& p, const Mv& m) {
    return (p.bb[KING] & bit(m.from)) && std::abs((m.to & 7) - (m.from & 7)) == 2 &&
           (m.from >> 3) == (m.to >> 3);
}

// 着法走完后己方王是否安全（真合法性；不处理易位——易位在生成时已按 python-chess 条件检查）
bool safe_after(const Pos& p, const Mv& m) {
    Pos q = p;
    int us = p.turn;
    int t = p.type_at(m.from);
    bool ep_cap = t == PAWN && m.to == p.ep && !(p.occ() & bit(m.to)) &&
                  (std::abs(m.to - m.from) == 7 || std::abs(m.to - m.from) == 9);
    q.remove(m.from);
    if (ep_cap) q.remove(m.to + (us == WHITE ? -8 : 8));
    q.put(m.to, m.promo ? m.promo : t, us);
    BB k = q.bb[KING] & q.co[us];
    if (!k) return true;
    return attackers(q, us ^ 1, msb(k), q.occ()) == 0;
}

void gen_legal(const Pos& p, std::vector<Mv>& out) {
    out.clear();
    std::vector<Mv> ps;
    ps.reserve(64);
    gen_pseudo(p, ps);
    BB kmask = p.bb[KING] & p.co[p.turn];
    if (!kmask) {
        out = ps;
        return;
    }
    int ksq = msb(kmask);
    bool check = attackers(p, p.turn ^ 1, ksq, p.occ()) != 0;
    if (!check) {
        for (const Mv& m : ps)
            if (is_castle_mv(p, m) || safe_after(p, m)) out.push_back(m);
        return;
    }
    // 被将军：python-chess 先出王的躲避着法（to 降序），再按常规顺序出其余着法
    for (const Mv& m : ps)
        if (m.from == ksq && !is_castle_mv(p, m) && safe_after(p, m)) out.push_back(m);
    for (const Mv& m : ps)
        if (m.from != ksq && safe_after(p, m)) out.push_back(m);
}

bool has_legal_ep(const Pos& p) {
    if (p.ep < 0) return false;
    std::vector<Mv> e;
    gen_ep(p, e);
    for (const Mv& m : e)
        if (safe_after(p, m)) return true;
    return false;
}

// ---------- push / 置换键 / 重复 ----------

struct Key {
    BB bb[7];
    BB co[2];
    BB castling;
    int turn;
    int ep;
    bool operator==(const Key& o) const {
        for (int t = PAWN; t <= KING; ++t)
            if (bb[t] != o.bb[t]) return false;
        return co[0] == o.co[0] && co[1] == o.co[1] && castling == o.castling && turn == o.turn &&
               ep == o.ep;
    }
};

Key key_of(const Pos& p) {
    Key k;
    std::memcpy(k.bb, p.bb, sizeof(k.bb));
    k.bb[0] = 0;
    k.co[0] = p.co[0];
    k.co[1] = p.co[1];
    k.castling = p.castling;
    k.turn = p.turn;
    k.ep = has_legal_ep(p) ? p.ep : -1;
    return k;
}

// python-chess is_irreversible（走子前的局面上求值；move 用标准写法，易位是 e1g1）
bool irreversible(const Pos& p, const Mv& m) {
    BB touched = bit(m.from) ^ bit(m.to);
    if ((touched & p.bb[PAWN]) || (touched & p.co[p.turn ^ 1])) return true;
    BB cr = p.castling;
    if (touched & cr) return true;
    if ((cr & RANK_1) && (touched & p.bb[KING] & p.co[WHITE])) return true;
    if ((cr & RANK_8) && (touched & p.bb[KING] & p.co[BLACK])) return true;
    return has_legal_ep(p);
}

Pos push(const Pos& p0, Mv m) {
    Pos p = p0;
    int us = p.turn;
    // _to_chess960：王从 e 线走到 g/c 线且目标格没有车 → 视为吃己方车的易位
    if (m.from == 4 && (p.bb[KING] & bit(4))) {
        if (m.to == 6 && !(p.bb[ROOK] & bit(6))) m.to = 7;
        else if (m.to == 2 && !(p.bb[ROOK] & bit(2))) m.to = 0;
    } else if (m.from == 60 && (p.bb[KING] & bit(60))) {
        if (m.to == 62 && !(p.bb[ROOK] & bit(62))) m.to = 63;
        else if (m.to == 58 && !(p.bb[ROOK] & bit(58))) m.to = 56;
    }
    int ep_prev = p.ep;
    p.ep = -1;
    p.hmc += 1;
    BB touched = bit(m.from) ^ bit(m.to);
    if ((touched & p.bb[PAWN]) || (touched & p.co[us ^ 1])) p.hmc = 0;
    BB from_bb = bit(m.from), to_bb = bit(m.to);
    int piece = p.type_at(m.from);
    p.remove(m.from);
    int captured = p.type_at(m.to);
    p.castling &= ~to_bb & ~from_bb;
    if (piece == KING) {
        p.castling &= us == WHITE ? ~RANK_1 : ~RANK_8;
    } else if (captured == KING) {
        if (us == WHITE && (m.to >> 3) == 7) p.castling &= ~RANK_8;
        else if (us == BLACK && (m.to >> 3) == 0) p.castling &= ~RANK_1;
    }
    if (piece == PAWN) {
        int diff = m.to - m.from;
        if (diff == 16 && (m.from >> 3) == 1) p.ep = m.from + 8;
        else if (diff == -16 && (m.from >> 3) == 6) p.ep = m.from - 8;
        else if (m.to == ep_prev && (std::abs(diff) == 7 || std::abs(diff) == 9) && !captured)
            p.remove(ep_prev + (us == WHITE ? -8 : 8));
    }
    if (m.promo) piece = m.promo;
    bool castle = piece == KING && (p.co[us] & to_bb);
    if (castle) {
        bool a_side = (m.to & 7) < (m.from & 7);
        p.remove(m.from);
        p.remove(m.to);
        int rank = us == WHITE ? 0 : 56;
        if (a_side) {
            p.put(rank + 2, KING, us);
            p.put(rank + 3, ROOK, us);
        } else {
            p.put(rank + 6, KING, us);
            p.put(rank + 5, ROOK, us);
        }
    } else {
        p.put(m.to, piece, us);
    }
    p.turn = us ^ 1;
    return p;
}

struct Hist {
    Key key;   // 该局面的置换键
    bool irr;  // 从该局面走出的那步是否不可逆
};

// python-chess is_repetition(count)：hist[0..n) 是当前局面之前的全部局面
bool is_repetition(const std::vector<Hist>& hist, size_t n, const Key& cur, int count) {
    size_t j = n;
    while (true) {
        if (count <= 1) return true;
        if (j < (size_t)(count - 1)) return false;
        --j;
        if (hist[j].irr) return false;
        if (hist[j].key == cur) --count;
    }
}

bool insufficient_color(const Pos& p, int c) {
    BB own = p.co[c];
    if (own & (p.bb[PAWN] | p.bb[ROOK] | p.bb[QUEEN])) return false;
    if (own & p.bb[KNIGHT])
        return popcnt(own) <= 2 && !(p.co[c ^ 1] & ~p.bb[KING] & ~p.bb[QUEEN]);
    if (own & p.bb[BISHOP]) {
        bool same = !(p.bb[BISHOP] & DARK_SQUARES) || !(p.bb[BISHOP] & LIGHT_SQUARES);
        return same && !p.bb[PAWN] && !p.bb[KNIGHT];
    }
    return true;
}

// Kit PUCT.exact_value 的棋规部分（不含残局表）。返回 true 表示有精确值。
bool rules_value(const Pos& p, const std::vector<Hist>& hist, size_t n, const std::vector<Mv>& legal,
                 bool claim_draw, double* v) {
    bool check = is_check(p);
    if (check && legal.empty()) {
        *v = -1.0;
        return true;
    }
    if (claim_draw) {
        Key k = key_of(p);
        if (is_repetition(hist, n, k, 3) || (p.hmc >= 100 && !legal.empty())) {
            *v = 0.0;
            return true;
        }
    }
    bool draw = (!check && legal.empty()) ||
                (insufficient_color(p, WHITE) && insufficient_color(p, BLACK)) ||
                (p.hmc >= 150 && !legal.empty());
    if (!draw) draw = is_repetition(hist, n, key_of(p), 5);
    if (draw) {
        *v = 0.0;
        return true;
    }
    return false;
}

bool has_side_castling(const Pos& p, int color, bool kingside) {
    BB backrank = color == WHITE ? RANK_1 : RANK_8;
    BB king_mask = p.bb[KING] & p.co[color] & backrank;
    if (!king_mask) return false;
    BB cr = p.castling & backrank;
    while (cr) {
        BB rook = cr & (~cr + 1);
        if (kingside ? rook > king_mask : rook < king_mask) return true;
        cr &= cr - 1;
    }
    return false;
}

const int PLANES = 19, PLANE_SIZE = PLANES * 64;

// contrib/planes19.encode：行棋方视角（黑方行棋时上下镜像并交换颜色）
void encode(const Pos& p, int reps, float* out) {
    std::memset(out, 0, sizeof(float) * PLANE_SIZE);
    int us = p.turn;
    auto orient = [us](int sq) { return us == WHITE ? sq : (sq ^ 56); };
    for (int t = PAWN; t <= KING; ++t) {
        BB b = p.bb[t];
        while (b) {
            int s = lsb(b);
            b &= b - 1;
            int color = (p.co[WHITE] & bit(s)) ? WHITE : BLACK;
            int plane = (t - 1) + (color == us ? 0 : 6);
            out[plane * 64 + orient(s)] = 1.0f;
        }
    }
    auto fill = [out](int plane, float v) {
        for (int i = 0; i < 64; ++i) out[plane * 64 + i] = v;
    };
    if (has_side_castling(p, us, true)) fill(12, 1.0f);
    if (has_side_castling(p, us, false)) fill(13, 1.0f);
    if (has_side_castling(p, us ^ 1, true)) fill(14, 1.0f);
    if (has_side_castling(p, us ^ 1, false)) fill(15, 1.0f);
    if (p.ep >= 0) out[16 * 64 + orient(p.ep)] = 1.0f;
    fill(17, (float)((double)std::min(p.hmc, 100) / 100.0));
    fill(18, (float)((double)std::min(reps, 2) / 2.0));
}

int repetitions_of(const Pos& p, const std::vector<Hist>& hist, size_t n) {
    Key k = key_of(p);
    return is_repetition(hist, n, k, 3) ? 2 : (is_repetition(hist, n, k, 2) ? 1 : 0);
}

std::string fen_of(const Pos& p) {
    static const char* PC = " pnbrqk";
    std::string s;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int sq = r * 8 + f;
            int t = p.type_at(sq);
            if (!t) {
                ++empty;
                continue;
            }
            if (empty) s += (char)('0' + empty);
            empty = 0;
            char c = PC[t];
            if (p.co[WHITE] & bit(sq)) c = (char)(c - 'a' + 'A');
            s += c;
        }
        if (empty) s += (char)('0' + empty);
        if (r) s += '/';
    }
    s += p.turn == WHITE ? " w " : " b ";
    std::string cr;
    if (p.castling & bit(7)) cr += 'K';
    if (p.castling & bit(0)) cr += 'Q';
    if (p.castling & bit(63)) cr += 'k';
    if (p.castling & bit(56)) cr += 'q';
    s += cr.empty() ? "-" : cr;
    s += ' ';
    if (p.ep >= 0) {
        s += (char)('a' + (p.ep & 7));
        s += (char)('1' + (p.ep >> 3));
    } else {
        s += '-';
    }
    s += ' ' + std::to_string(p.hmc) + " 1";
    return s;
}

// ---------- numpy 数值语义 ----------

// numpy pairwise 求和（loops_utils.h：<8 顺序累加；<=128 八路展开；否则对半递归，分点对齐 8）
template <typename T>
T pairwise_sum(const T* a, size_t n) {
    if (n < 8) {
        T res = 0;
        for (size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        T r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// ---------- 搜索树 ----------

struct Node;
typedef std::shared_ptr<Node> NodeP;

struct Node {
    std::vector<uint16_t> moves;
    std::vector<float> P, W, VL;
    std::vector<int32_t> N;
    std::vector<NodeP> children;
    bool expanded = false;
    bool has_term = false;
    double term = 0.0;
    int64_t sum_N = 0;

    void expand(const std::vector<Mv>& mvs, const std::vector<float>& pri) {
        size_t n = mvs.size();
        moves.resize(n);
        for (size_t i = 0; i < n; ++i) moves[i] = mv_code(mvs[i]);
        P = pri;
        N.assign(n, 0);
        W.assign(n, 0.0f);
        VL.assign(n, 0.0f);
        children.assign(n, NodeP());
        expanded = true;
    }
};

struct Leaf {
    Node* node;
    std::vector<std::pair<Node*, int>> path;
    Pos pos;
    std::vector<Mv> legal;
};

struct Ctx {
    // PUCTConfig
    double c_base, c_init, fpu_reduction, vl;
    bool claim_draw;
    int max_collision, root_min_visits, oracle_max_pieces;
    bool repetition_draw = true;   // 走到已出现过的局面按和棋（值 0）结算，不再下探
    // 根局面与对局历史
    Pos root_pos;
    std::vector<Hist> hist;  // 前 root_n 项是对局历史，下探时临时追加路径
    size_t root_n = 0;
    NodeP root;
    // 一次收集的状态（可因残局表探测暂停）
    bool collecting = false;
    int want = 0, terminal_sims = 0, spins = 0, max_depth = 0;
    int64_t collisions = 0;
    std::unordered_set<Node*> pending;
    std::vector<Leaf> leaves;
    bool probe_wait = false, probe_answered = false, probe_has = false;
    double probe_value = 0.0;
    Leaf probe;
    // best_child 的临时缓冲
    std::vector<double> denom, vis64;
    std::vector<float> num, vis32;
};

thread_local std::string g_err;

int best_child(Ctx& c, const Node& n) {
    size_t k = n.moves.size();
    int64_t total_i = std::max<int64_t>(n.sum_N, 1);
    double cc = std::log(((double)(1 + total_i) + c.c_base) / c.c_base) + c.c_init;
    float vl32 = (float)c.vl;
    c.denom.resize(k);
    c.num.resize(k);
    c.vis32.clear();
    c.vis64.clear();
    for (size_t a = 0; a < k; ++a) {
        c.denom[a] = (double)n.N[a] + (double)n.VL[a];
        float t = vl32 * n.VL[a];
        c.num[a] = n.W[a] - t;
        if (c.denom[a] > 0) {
            c.vis32.push_back(c.num[a]);
            c.vis64.push_back(c.denom[a]);
        }
    }
    double parent_q = 0.0;
    if (!c.vis32.empty())
        parent_q = (double)pairwise_sum(c.vis32.data(), c.vis32.size()) /
                   pairwise_sum(c.vis64.data(), c.vis64.size());
    float q_unvisited = (float)(parent_q - c.fpu_reduction);
    float c32 = (float)cc;
    float s32 = (float)std::sqrt((double)total_i);
    int best = -1;
    double best_s = 0.0;
    for (size_t a = 0; a < k; ++a) {
        float q = c.denom[a] > 0 ? (float)((double)c.num[a] / c.denom[a]) : q_unvisited;
        float u32 = (c32 * n.P[a]) * s32;
        double s = (double)q + (double)u32 / (1.0 + c.denom[a]);
        if (best < 0 || s > best_s) {
            best = (int)a;
            best_s = s;
        }
    }
    return best;
}

void backup(Ctx& c, const std::vector<std::pair<Node*, int>>& path, double value) {
    double v = value;
    float vl32 = (float)c.vl;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        v = -v;
        Node* n = it->first;
        int i = it->second;
        n->N[i] += 1;
        n->W[i] = n->W[i] + (float)v;
        n->VL[i] = n->VL[i] - vl32;
        n->sum_N += 1;
    }
}

void revert_vl(Ctx& c, const std::vector<std::pair<Node*, int>>& path) {
    float vl32 = (float)c.vl;
    for (auto& e : path) e.first->VL[e.second] = e.first->VL[e.second] - vl32;
}

// contrib/planes19.priors_from_policy
void priors_of(const Pos& p, const std::vector<Mv>& legal, const float* policy, const float* promo,
               std::vector<float>& out) {
    size_t n = legal.size();
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const Mv& m = legal[i];
        int f = m.from, t = m.to;
        if (p.turn == BLACK) {
            f ^= 56;
            t ^= 56;
        }
        double s = (double)policy[f * 64 + t];
        if (m.promo) s *= (double)promo[QUEEN - m.promo];  // PROMO_PIECES = (Q, R, B, N)
        out[i] = (float)s;
    }
    float total = pairwise_sum(out.data(), n);
    if (total <= 0) {
        float u = (float)(1.0 / (double)n);
        for (size_t i = 0; i < n; ++i) out[i] = u;
    } else {
        for (size_t i = 0; i < n; ++i) out[i] = out[i] / total;
    }
}

// 叶子确定要送网络：撞上待定叶子则只撤销 virtual loss；否则登记并编码
void finish_leaf(Ctx& c, Leaf& lf, float* planes) {
    if (c.pending.count(lf.node)) {
        c.collisions += 1;
        revert_vl(c, lf.path);
        c.spins += 1;
        return;
    }
    c.pending.insert(lf.node);
    encode(lf.pos, repetitions_of(lf.pos, c.hist, c.hist.size()),
           planes + (size_t)c.leaves.size() * PLANE_SIZE);
    c.max_depth = std::max(c.max_depth, (int)lf.path.size());
    c.leaves.push_back(std::move(lf));
    c.spins += 1;
}

void set_terminal_and_backup(Ctx& c, Node* node, const std::vector<std::pair<Node*, int>>& path,
                             double v) {
    node->expanded = true;
    node->has_term = true;
    node->term = v;
    backup(c, path, v);
    c.terminal_sims += 1;
    c.spins += 1;
}

// 返回 >=0 叶子数（收集完成）；-1 需要残局表探测（路径写入 path_out）
int collect(Ctx& c, int want, float* planes, uint16_t* path_out, int path_cap, int* path_len) {
    if (!c.collecting) {
        c.collecting = true;
        c.want = want;
        c.terminal_sims = 0;
        c.spins = 0;
        c.max_depth = 0;
        c.pending.clear();
        c.leaves.clear();
    }
    if (c.probe_wait) {
        if (!c.probe_answered) throw std::runtime_error("残局表探测尚未回填（kp_probe_answer）");
        c.probe_wait = false;
        Leaf lf = std::move(c.probe);
        if (c.probe_has) set_terminal_and_backup(c, lf.node, lf.path, c.probe_value);
        else finish_leaf(c, lf, planes);
        c.hist.resize(c.root_n);
    }
    float vl32 = (float)c.vl;
    while ((int)c.leaves.size() + c.terminal_sims < c.want && c.spins < c.max_collision * c.want) {
        Node* node = c.root.get();
        Pos pos = c.root_pos;
        std::vector<std::pair<Node*, int>> path;
        bool first = true;
        while (node->expanded && !node->has_term) {
            if (node->moves.empty()) break;
            int i = -1;
            if (first && c.root_min_visits > 0) {
                for (size_t a = 0; a < node->moves.size(); ++a)
                    if ((double)node->N[a] + (double)node->VL[a] < (double)c.root_min_visits) {
                        i = (int)a;
                        break;
                    }
            }
            if (i < 0) i = best_child(c, *node);
            first = false;
            node->VL[i] = node->VL[i] + vl32;
            path.emplace_back(node, i);
            Mv m = mv_decode(node->moves[i]);
            c.hist.push_back(Hist{key_of(pos), irreversible(pos, m)});
            pos = push(pos, m);
            if (!node->children[i]) node->children[i] = std::make_shared<Node>();
            node = node->children[i].get();
            // 这一步走到本局（含搜索路径）里已出现过的局面：按和棋结算（AlphaZero 口径）。
            // 只标记不在此处回填：下面按 has_term 分支统一 backup，与 Python 版同构。
            if (c.repetition_draw && !node->has_term &&
                is_repetition(c.hist, c.hist.size(), key_of(pos), 2)) {
                node->expanded = true;
                node->has_term = true;
                node->term = 0.0;
            }
        }
        if (node->has_term) {
            backup(c, path, node->term);
            c.terminal_sims += 1;
            c.spins += 1;
            c.hist.resize(c.root_n);
            continue;
        }
        if (node->expanded) {
            backup(c, path, 0.0);
            c.terminal_sims += 1;
            c.spins += 1;
            c.hist.resize(c.root_n);
            continue;
        }
        Leaf lf{node, std::move(path), pos, {}};
        if (c.pending.count(node)) {  // 同一节点的精确值与上次相同（仍为 None），直接按碰撞处理
            finish_leaf(c, lf, planes);
            c.hist.resize(c.root_n);
            continue;
        }
        gen_legal(pos, lf.legal);
        double v;
        if (rules_value(pos, c.hist, c.hist.size(), lf.legal, c.claim_draw, &v)) {
            set_terminal_and_backup(c, node, lf.path, v);
            c.hist.resize(c.root_n);
            continue;
        }
        if (c.oracle_max_pieces >= 0 && popcnt(pos.occ()) <= c.oracle_max_pieces) {
            // 交给 Python 探测：回传从根出发的着法序列，Python 端在原棋盘（含走子栈）上重放后探测
            int len = (int)lf.path.size();
            if (len > path_cap) throw std::runtime_error("路径缓冲区太小");
            for (int k = 0; k < len; ++k) path_out[k] = lf.path[k].first->moves[lf.path[k].second];
            *path_len = len;
            c.probe = std::move(lf);
            c.probe_wait = true;
            c.probe_answered = false;
            return -1;  // hist 保留路径，回填后 finish_leaf 编码要用
        }
        finish_leaf(c, lf, planes);
        c.hist.resize(c.root_n);
    }
    c.collecting = false;
    return (int)c.leaves.size();
}

void apply(Ctx& c, const float* policy, const float* promo, const float* wdl, int n) {
    if (c.collecting || c.probe_wait) throw std::runtime_error("收集尚未完成就调用了 apply");
    if (n != (int)c.leaves.size())
        throw std::runtime_error("apply 的结果数与叶子数不符：" + std::to_string(n) + " != " +
                                 std::to_string(c.leaves.size()));
    std::vector<float> pri;
    for (int k = 0; k < n; ++k) {
        Leaf& lf = c.leaves[k];
        if (lf.legal.empty()) {
            lf.node->expanded = true;
            lf.node->has_term = true;
            lf.node->term = 0.0;
            backup(c, lf.path, 0.0);
            continue;
        }
        priors_of(lf.pos, lf.legal, policy + (size_t)k * 4096, promo + (size_t)k * 4, pri);
        lf.node->expand(lf.legal, pri);
        float value = wdl[k * 3 + 0] - wdl[k * 3 + 2];
        backup(c, lf.path, (double)value);
    }
    c.leaves.clear();
    c.pending.clear();
}

bool load_root(Ctx& c, const uint64_t* bbs, int turn, uint64_t castling, int ep, int hmc,
               const uint16_t* moves, int n_moves) {
    Pos p;
    std::memset(&p, 0, sizeof(p));
    for (int t = PAWN; t <= KING; ++t) p.bb[t] = bbs[t - 1];
    p.co[WHITE] = bbs[6];
    p.co[BLACK] = bbs[7];
    p.turn = turn ? WHITE : BLACK;
    p.castling = clean_castling(p, castling);
    p.ep = ep;
    p.hmc = hmc;
    c.hist.clear();
    std::vector<Mv> legal;
    for (int i = 0; i < n_moves; ++i) {
        Mv m = mv_decode(moves[i]);
        gen_legal(p, legal);
        bool ok = false;
        for (const Mv& x : legal)
            if (x.from == m.from && x.to == m.to && x.promo == m.promo) ok = true;
        if (!ok) {
            g_err = "对局历史第 " + std::to_string(i) + " 步不是合法着法";
            return false;
        }
        c.hist.push_back(Hist{key_of(p), irreversible(p, m)});
        p = push(p, m);
    }
    c.root_pos = p;
    c.root_n = c.hist.size();
    return true;
}

}  // namespace

// ---------- C ABI ----------

#define KP_API extern "C" __attribute__((visibility("default")))
#define KP_TRY try {
#define KP_CATCH(ret)                 \
    }                                 \
    catch (const std::exception& e) { \
        g_err = e.what();             \
        return ret;                   \
    }                                 \
    catch (...) {                     \
        g_err = "unknown C++ error";  \
        return ret;                   \
    }

KP_API int kp_abi_version() { return 1; }
KP_API const char* kp_last_error() { return g_err.c_str(); }

KP_API void* kp_ctx_new(double c_base, double c_init, double fpu_reduction, double vl, int claim_draw,
                        int max_collision, int root_min_visits, int oracle_max_pieces,
                        int repetition_draw) {
    KP_TRY
    Ctx* c = new Ctx();
    c->c_base = c_base;
    c->c_init = c_init;
    c->fpu_reduction = fpu_reduction;
    c->vl = vl;
    c->claim_draw = claim_draw != 0;
    c->max_collision = max_collision;
    c->root_min_visits = root_min_visits;
    c->oracle_max_pieces = oracle_max_pieces;
    c->repetition_draw = repetition_draw != 0;
    return c;
    KP_CATCH(nullptr)
}

KP_API void kp_ctx_free(void* ctx) { delete (Ctx*)ctx; }

KP_API int kp_set_root(void* ctx, const uint64_t* bbs, int turn, uint64_t castling, int ep, int hmc,
                       const uint16_t* moves, int n_moves) {
    KP_TRY
    Ctx& c = *(Ctx*)ctx;
    if (c.collecting || c.probe_wait) throw std::runtime_error("上一次收集尚未完成");
    return load_root(c, bbs, turn, castling, ep, hmc, moves, n_moves) ? 0 : -2;
    KP_CATCH(-2)
}

KP_API int kp_begin(void* ctx, void* node) {
    KP_TRY
    Ctx& c = *(Ctx*)ctx;
    c.root = node ? *(NodeP*)node : NodeP();  // NULL = 搜索结束，放开对树的引用
    c.collecting = false;
    c.probe_wait = false;
    c.leaves.clear();
    c.pending.clear();
    c.collisions = 0;
    c.hist.resize(c.root_n);
    return 0;
    KP_CATCH(-2)
}

KP_API int kp_encode_root(void* ctx, float* out) {
    KP_TRY
    Ctx& c = *(Ctx*)ctx;
    encode(c.root_pos, repetitions_of(c.root_pos, c.hist, c.root_n), out);
    return 0;
    KP_CATCH(-2)
}

// 根节点展开（不回传）；返回合法着法数，0 表示无着法（按和棋终局处理）
KP_API int kp_expand_root(void* ctx, const float* policy, const float* promo, const float* wdl) {
    KP_TRY
    (void)wdl;
    Ctx& c = *(Ctx*)ctx;
    std::vector<Mv> legal;
    gen_legal(c.root_pos, legal);
    Node* r = c.root.get();
    if (legal.empty()) {
        r->expanded = true;
        r->has_term = true;
        r->term = 0.0;
        return 0;
    }
    std::vector<float> pri;
    priors_of(c.root_pos, legal, policy, promo, pri);
    r->expand(legal, pri);
    return (int)legal.size();
    KP_CATCH(-2)
}

// info[0] = 本次收集里终局 / 残局表结算的模拟数，info[1] = 叶子最大深度，
// info[2] = 本次搜索累计碰撞数，info[3] = 待探测路径长度（仅返回 -1 时有效）
KP_API int kp_collect(void* ctx, int want, float* planes, int* info, uint16_t* path_out,
                      int path_cap) {
    KP_TRY
    Ctx& c = *(Ctx*)ctx;
    info[3] = 0;
    int n = collect(c, want, planes, path_out, path_cap, &info[3]);
    info[0] = c.terminal_sims;
    info[1] = c.max_depth;
    info[2] = (int)c.collisions;
    return n;
    KP_CATCH(-2)
}

KP_API int kp_probe_answer(void* ctx, int has_value, double value) {
    KP_TRY
    Ctx& c = *(Ctx*)ctx;
    if (!c.probe_wait) throw std::runtime_error("没有待回填的残局表探测");
    c.probe_answered = true;
    c.probe_has = has_value != 0;
    c.probe_value = value;
    return 0;
    KP_CATCH(-2)
}

KP_API int kp_apply(void* ctx, const float* policy, const float* promo, const float* wdl, int n) {
    KP_TRY
    apply(*(Ctx*)ctx, policy, promo, wdl, n);
    return 0;
    KP_CATCH(-2)
}

KP_API int64_t kp_collisions(void* ctx) { return ((Ctx*)ctx)->collisions; }

// ---- 节点句柄（堆上的 shared_ptr：Python 持有期间子树不释放，与 Python 版的 GC 语义一致）----

KP_API void* kp_node_new() {
    KP_TRY
    return new NodeP(std::make_shared<Node>());
    KP_CATCH(nullptr)
}

KP_API void kp_node_free(void* h) { delete (NodeP*)h; }

KP_API int kp_node_state(void* h, int* expanded, int* has_term, double* term, int64_t* sum_N) {
    const Node& n = **(NodeP*)h;
    *expanded = n.expanded;
    *has_term = n.has_term;
    *term = n.term;
    *sum_N = n.sum_N;
    return (int)n.moves.size();
}

KP_API void kp_node_arrays(void* h, uint16_t* moves, float* P, int32_t* N, float* W, float* VL) {
    const Node& n = **(NodeP*)h;
    size_t k = n.moves.size();
    if (moves) std::memcpy(moves, n.moves.data(), k * sizeof(uint16_t));
    if (P) std::memcpy(P, n.P.data(), k * sizeof(float));
    if (N) std::memcpy(N, n.N.data(), k * sizeof(int32_t));
    if (W) std::memcpy(W, n.W.data(), k * sizeof(float));
    if (VL) std::memcpy(VL, n.VL.data(), k * sizeof(float));
}

KP_API int kp_node_set_P(void* h, const float* P, int n) {
    Node& node = **(NodeP*)h;
    if (n != (int)node.moves.size()) {
        g_err = "P 的长度与着法数不符";
        return -2;
    }
    std::memcpy(node.P.data(), P, (size_t)n * sizeof(float));
    return 0;
}

KP_API void kp_node_set_terminal(void* h, double v) {
    Node& n = **(NodeP*)h;
    n.expanded = true;
    n.has_term = true;
    n.term = v;
}

// 第 i 个子节点的句柄；不存在返回 NULL
KP_API void* kp_node_child(void* h, int i) {
    Node& n = **(NodeP*)h;
    if (i < 0 || i >= (int)n.children.size() || !n.children[i]) return nullptr;
    return new NodeP(n.children[i]);
}

// PUCT.advance_root：找到着法对应的子节点、清零其 virtual loss；没搜到返回 NULL
KP_API void* kp_node_advance(void* h, int code) {
    Node& n = **(NodeP*)h;
    if (!n.expanded) return nullptr;
    for (size_t i = 0; i < n.moves.size(); ++i) {
        if (n.moves[i] != (uint16_t)code) continue;
        NodeP ch = n.children[i];
        if (!ch) return nullptr;
        std::fill(ch->VL.begin(), ch->VL.end(), 0.0f);
        return new NodeP(ch);
    }
    return nullptr;
}

// PUCT.principal_variation
KP_API int kp_node_pv(void* h, int max_len, uint16_t* out) {
    const Node* n = ((NodeP*)h)->get();
    int len = 0;
    while (n && !n->moves.empty() && len < max_len) {
        size_t j = 0;
        for (size_t a = 1; a < n->N.size(); ++a)
            if (n->N[a] > n->N[j]) j = a;
        if (n->N[j] <= 0) break;
        out[len++] = n->moves[j];
        n = n->children[j].get();
    }
    return len;
}

// ---- 测试钩子：根局面上的棋规结果（与 python-chess 对照）----

KP_API int kp_root_legal(void* ctx, uint16_t* out, int cap) {
    KP_TRY
    std::vector<Mv> legal;
    gen_legal(((Ctx*)ctx)->root_pos, legal);
    int n = (int)legal.size();
    for (int i = 0; i < n && i < cap; ++i) out[i] = mv_code(legal[i]);
    return n;
    KP_CATCH(-2)
}

// 返回 1 有精确值（写入 *v），0 没有；reps[0..3] = is_repetition(2/3/4/5)，info[0] = is_check
KP_API int kp_root_rules(void* ctx, int claim_draw, double* v, int* reps, int* info) {
    KP_TRY
    Ctx& c = *(Ctx*)ctx;
    std::vector<Mv> legal;
    gen_legal(c.root_pos, legal);
    Key k = key_of(c.root_pos);
    for (int i = 0; i < 4; ++i) reps[i] = is_repetition(c.hist, c.root_n, k, i + 2);
    info[0] = is_check(c.root_pos);
    info[1] = has_legal_ep(c.root_pos);
    return rules_value(c.root_pos, c.hist, c.root_n, legal, claim_draw != 0, v) ? 1 : 0;
    KP_CATCH(-2)
}

KP_API int kp_root_fen(void* ctx, char* out, int cap) {
    KP_TRY
    std::string f = fen_of(((Ctx*)ctx)->root_pos);
    if ((int)f.size() + 1 > cap) return -2;
    std::memcpy(out, f.c_str(), f.size() + 1);
    return (int)f.size();
    KP_CATCH(-2)
}

// 对 float32 数组做 numpy 口径的 pairwise 求和（测试钩子）
KP_API float kp_pairwise_f32(const float* a, int n) { return pairwise_sum(a, (size_t)n); }
KP_API double kp_pairwise_f64(const double* a, int n) { return pairwise_sum(a, (size_t)n); }

// ==================== Gumbel 顺序减半：节点级算术 ====================
//
// 契约：与 ``Kit/search/gumbel.py`` 的对应函数**数值等价**（非逐位一致）——
// numpy 的 fp32 exp 与 BLAS dot 跟 libm / 手写累加有 1 ULP 级差异，因此：
//   - ``kg_select_action`` 与 Python ``select_action`` 的决策一致率须 ≥ 99.9%
//     （只在近-平局处允许不同）；``kg_pi_prime`` 的 max|Δ| ≤ 1e-5。
// 对照 battery：tests/test_gumbel_cpp.py。
//
// 宽度/次序按 numpy 2.x（NEP 50）抄写：Python float 是弱标量，与 fp32 数组/标量运算
// 保持 fp32；``N`` 是 int64，凡与它混合处都显式 ``(float)`` 转换以复刻
// ``n.astype(np.float32)``。根 Gumbel 噪声由 Python 采样（``rng.random(dtype=fp32)`` →
// ``-log(-log(u+1e-20))``），C++ 只消费给定数组 —— 不移植 PCG64。

namespace {

// numpy softmax：max 在 fp32、exp 在 fp32、和用 float64 pairwise、除法 fp32
void kg_softmax_into(const float* x, int n, float* out) {
    float mx = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] > mx) mx = x[i];
    std::vector<double> acc((size_t)n);
    for (int i = 0; i < n; ++i) {
        out[i] = std::exp(x[i] - mx);
        acc[(size_t)i] = (double)out[i];
    }
    const float fs = (float)pairwise_sum(acc.data(), (size_t)n);
    for (int i = 0; i < n; ++i) out[i] = out[i] / fs;
}

// v_mix（gumbel.py:132）：pi 必须是 softmax(logits) 的同一份数值
double kg_vmix(const float* pi, const int64_t* N, const float* QSUM, int n, double q) {
    int64_t n_tot = 0;
    for (int i = 0; i < n; ++i) n_tot += N[i];
    if (n <= 0 || n_tot == 0) return q;
    std::vector<float> pv, qv;
    for (int i = 0; i < n; ++i) {
        if (N[i] > 0) {
            pv.push_back(pi[i]);
            qv.push_back(QSUM[i] / (float)N[i]);
        }
    }
    double num = 0.0;  // np.dot(pv, qv)：BLAS 求和顺序不可复刻，用 fp64 累加近似
    for (size_t i = 0; i < pv.size(); ++i) num += (double)pv[i] * (double)qv[i];
    const double den = (double)pairwise_sum(pv.data(), pv.size()) + 1e-8;
    return (q + (double)n_tot * (num / den)) / (1.0 + (double)n_tot);
}

// completedQ（gumbel.py:145）
void kg_completed_q(const float* pi, const int64_t* N, const float* QSUM, int n, double q,
                    float* out) {
    const float vmf = (float)kg_vmix(pi, N, QSUM, n, q);
    for (int i = 0; i < n; ++i) out[i] = vmf;
    for (int i = 0; i < n; ++i)
        if (N[i] > 0) out[i] = QSUM[i] / (float)N[i];
}

// σ(q̂)（gumbel.py:157）：q̂ 用本节点 completed Q 的量程归一
void kg_qtransform(const float* pi, const int64_t* N, const float* QSUM, int n, double q,
                   double c_visit, double c_scale, float* out) {
    kg_completed_q(pi, N, QSUM, n, q, out);
    if (n <= 0) return;
    float qmin = out[0], qmax = out[0];
    int64_t nmax = 0;
    for (int i = 0; i < n; ++i) {
        if (out[i] < qmin) qmin = out[i];
        if (out[i] > qmax) qmax = out[i];
        if (N[i] > nmax) nmax = N[i];
    }
    const float span = (float)(((double)qmax - (double)qmin) + 1e-8);
    const float fv = (float)c_visit + (float)nmax;
    const float cs = (float)c_scale;
    for (int i = 0; i < n; ++i)
        out[i] = fv * cs * ((out[i] - qmin) / span);
}

// select_action 快路径（gumbel.py:180-215）：非根确定性选择
int kg_select_action_impl(const float* logits, int n, const int64_t* N, const float* QSUM,
                          double q, double c_visit, double c_scale, int* out) {
    if (n <= 0) return -2;
    std::vector<float> pi((size_t)n);
    kg_softmax_into(logits, n, pi.data());
    int64_t n_total = 0;
    for (int i = 0; i < n; ++i) n_total += N[i];
    int best = 0;
    if (n_total == 0) {
        // completed Q 全 = v̂ ⇒ σ ≡ 0 ⇒ π_imp = π；frac = N/1（全 0 ⇒ 不减）
        for (int i = 1; i < n; ++i)
            if (pi[i] - ((float)N[i] / 1.0f) > pi[best] - ((float)N[best] / 1.0f)) best = i;
        *out = best;
        return 0;
    }
    std::vector<float> qv, pv;
    for (int i = 0; i < n; ++i) {
        if (N[i] > 0) {
            qv.push_back(QSUM[i] / (float)N[i]);
            pv.push_back(pi[i]);
        }
    }
    double num = 0.0;
    for (size_t i = 0; i < pv.size(); ++i) num += (double)pv[i] * (double)qv[i];
    const double den = (double)pairwise_sum(pv.data(), pv.size()) + 1e-8;
    const double n_tot = (double)n_total;
    const double vm = (q + n_tot * (num / den)) / (1.0 + n_tot);
    std::vector<float> cq((size_t)n, (float)vm);
    for (int i = 0, k = 0; i < n; ++i)
        if (N[i] > 0) cq[i] = qv[(size_t)k++];
    float qmin = cq[0], qmax = cq[0];
    int64_t nmax = 0;
    for (int i = 0; i < n; ++i) {
        if (cq[i] < qmin) qmin = cq[i];
        if (cq[i] > qmax) qmax = cq[i];
        if (N[i] > nmax) nmax = N[i];
    }
    const float span = (float)(((double)qmax - (double)qmin) + 1e-8);
    const float fv = (float)c_visit + (float)nmax;
    const float cs = (float)c_scale;
    std::vector<float> imp((size_t)n);
    for (int i = 0; i < n; ++i)
        imp[i] = logits[i] + (fv * cs * ((cq[i] - qmin) / span));
    kg_softmax_into(imp.data(), n, imp.data());
    const float fn = (float)(1 + n_total);
    for (int i = 1; i < n; ++i)
        if (imp[i] - ((float)N[i] / fn) > imp[best] - ((float)N[best] / fn)) best = i;
    *out = best;
    return 0;
}

// π′ = softmax(ℓ + σ(completedQ))（gumbel.py:175）
void kg_pi_prime_impl(const float* logits, int n, const int64_t* N, const float* QSUM, double q,
                      double c_visit, double c_scale, float* out) {
    std::vector<float> pi((size_t)n), s((size_t)n), z((size_t)n);
    kg_softmax_into(logits, n, pi.data());
    kg_qtransform(pi.data(), N, QSUM, n, q, c_visit, c_scale, s.data());
    for (int i = 0; i < n; ++i) z[i] = logits[i] + s[i];
    kg_softmax_into(z.data(), n, out);
}

// gumbel_topm 的排序语义（gumbel.py:227）：稳定 argsort(-(noise+ℓ))[:m]，noise 由 Python 给
int kg_topm_impl(const float* logits, int n, const float* noise, int m0, int* out) {
    std::vector<float> score((size_t)n);
    for (int i = 0; i < n; ++i) score[i] = noise[i] + logits[i];
    std::vector<int> idx((size_t)n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return score[a] > score[b]; });
    const int m = std::min(m0, n);
    for (int i = 0; i < m; ++i) out[i] = idx[i];
    return m;
}

}  // namespace

KP_API int kg_softmax(const float* x, int n, float* out) {
    KP_TRY
    if (n <= 0) throw std::runtime_error("kg_softmax: n<=0");
    kg_softmax_into(x, n, out);
    return 0;
    KP_CATCH(-2)
}

KP_API int kg_select_action(const float* logits, int n, const int64_t* N, const float* QSUM,
                            double q, double c_visit, double c_scale, int* out) {
    KP_TRY
    return kg_select_action_impl(logits, n, N, QSUM, q, c_visit, c_scale, out);
    KP_CATCH(-2)
}

KP_API int kg_pi_prime(const float* logits, int n, const int64_t* N, const float* QSUM, double q,
                       double c_visit, double c_scale, float* out) {
    KP_TRY
    if (n <= 0) throw std::runtime_error("kg_pi_prime: n<=0");
    kg_pi_prime_impl(logits, n, N, QSUM, q, c_visit, c_scale, out);
    return 0;
    KP_CATCH(-2)
}

KP_API int kg_topm(const float* logits, int n, const float* noise, int m0, int* out) {
    KP_TRY
    return kg_topm_impl(logits, n, noise, m0, out);
    KP_CATCH(-2)
}

// ==================== Gumbel 顺序减半：搜索树 + 调度 ====================
//
// 与 ``Kit/search/gumbel.py`` 的 ``Gumbel`` / ``order_halving_gen`` 数值等价（非逐位，
// 契约见上）。按拍（wave）批量：一轮里每个存活候选各下探一次 = 一拍；一拍的叶子一次
// 交给 Python 前向（候选子树互不相交，批内顺序无关——与 Python parallel=True 的
// gather 同语义）。
//
// 状态机：kg_set_root → kg_expand_root(根前向) → kg_begin(噪声) → 循环
// [kg_collect 返回本拍需要前向的叶子数 → Python 前向 → kg_apply 回填] → 完成。
// 收集期间不允许再 collect（必须先 apply）。

namespace {

// 与 gumbel.py 的 NEG_LOGIT 一致：非法/零先验的 ℓ 取有限大负数
const float G_NEG_LOGIT = -3e4f;

struct GNode;
typedef std::shared_ptr<GNode> GNodeP;

struct GNode {
    std::vector<uint16_t> moves;
    std::vector<float> logits, QSUM;
    std::vector<int64_t> N;
    std::vector<GNodeP> children;
    std::vector<Hist> hist;   // 本局面之前的对局历史（根段 + 路径）
    Pos pos;
    double q = 0.0;           // 网络值 / 终局真值（均为行棋方视角）
    bool expanded = false;
    bool terminal = false;
    int depth = 0;
};

struct Pending {
    GNodeP node;                                    // 新展开的叶子
    std::vector<std::pair<GNode*, int>> path;       // 根→叶子的每一步 (父节点, 边)
};

struct GCtx {
    double c_visit = 50.0, c_scale = 0.1;
    int m0 = 16;
    bool claim_draw = true;
    std::vector<Hist> root_hist;
    Pos root_pos;
    GNodeP root;
    bool begun = false, finished = false, collecting = false;
    std::vector<int> cand;                 // 存活候选（根的边序号）
    std::vector<float> cand_noise;
    std::vector<int> rounds_budget;
    int round = 0, wave = 0;
    std::vector<int> ks;
    int sims_used = 0;
    int action = -1;
    std::vector<Pending> pending;
    int n_nodes = 0, n_terminal = 0, max_depth = 0;
};

// 把 priors 变成 ℓ：np.log 后 clamp 到 NEG_LOGIT（与 node_from_eval 一致）
void g_logits_of(const std::vector<float>& pri, std::vector<float>& out) {
    out.resize(pri.size());
    for (size_t i = 0; i < pri.size(); ++i)
        out[i] = std::max(std::log(pri[i]), G_NEG_LOGIT);
}

void g_fill_expanded(GNode* n, const std::vector<Mv>& legal, const std::vector<float>& pri) {
    size_t k = legal.size();
    n->moves.resize(k);
    for (size_t i = 0; i < k; ++i) n->moves[i] = mv_code(legal[i]);
    g_logits_of(pri, n->logits);
    n->N.assign(k, 0);
    n->QSUM.assign(k, 0.0f);
    n->children.assign(k, GNodeP());
    n->expanded = true;
}

// 展开 parent 的第 edge 个孩子。需要前向时登记 pending 并返回 nullptr（path 已含本步）。
GNodeP g_expand_child(GCtx& g, GNode* parent, int edge, float* planes, int& produced,
                      std::vector<std::pair<GNode*, int>>& path) {
    path.push_back({parent, edge});
    Mv mv = mv_decode(parent->moves[(size_t)edge]);
    Pos cp = push(parent->pos, mv);
    std::vector<Hist> ch = parent->hist;
    ch.push_back(Hist{key_of(parent->pos), irreversible(parent->pos, mv)});
    std::vector<Mv> legal;
    gen_legal(cp, legal);
    GNodeP nn = std::make_shared<GNode>();
    nn->pos = cp;
    nn->hist = std::move(ch);
    nn->depth = parent->depth + 1;
    if (nn->depth > g.max_depth) g.max_depth = nn->depth;
    double tv = 0.0;
    if (rules_value(cp, nn->hist, nn->hist.size(), legal, g.claim_draw, &tv)) {
        nn->terminal = true;
        nn->expanded = true;
        nn->q = tv;
        parent->children[(size_t)edge] = nn;
        g.n_nodes += 1;
        g.n_terminal += 1;
        return nn;
    }
    nn->moves.resize(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) nn->moves[i] = mv_code(legal[i]);
    parent->children[(size_t)edge] = nn;
    encode(cp, repetitions_of(cp, nn->hist, nn->hist.size()),
           planes + (size_t)produced * PLANE_SIZE);
    g.pending.push_back(Pending{nn, path});
    produced += 1;
    return nullptr;
}

// 非根下探（_simulate）：true = 有值（*out_val，node 视角）；false = 需要前向（pending 持有路径）
bool g_sim_node(GCtx& g, GNode* node, double* out_val, float* planes, int& produced,
                std::vector<std::pair<GNode*, int>>& path) {
    if (node->terminal) {
        *out_val = node->q;
        return true;
    }
    int a = -1;
    if (kg_select_action_impl(node->logits.data(), (int)node->moves.size(), node->N.data(),
                              node->QSUM.data(), node->q, g.c_visit, g.c_scale, &a) != 0)
        throw std::runtime_error("kg_select_action 失败");
    GNode* child = node->children[(size_t)a].get();
    double val;
    if (child == nullptr) {
        GNodeP nn = g_expand_child(g, node, a, planes, produced, path);
        if (!nn) return false;
        val = -nn->q;
    } else {
        path.push_back({node, a});
        double sub = 0.0;
        if (!g_sim_node(g, child, &sub, planes, produced, path)) return false;
        val = -sub;
    }
    node->N[(size_t)a] += 1;
    node->QSUM[(size_t)a] += (float)val;
    return true;
}

// 根的候选下探（_sim_root）：用候选动作 a（不走 select_action）
bool g_sim_root(GCtx& g, int a, float* planes, int& produced) {
    GNode* root = g.root.get();
    std::vector<std::pair<GNode*, int>> path;
    GNode* child = root->children[(size_t)a].get();
    double val;
    if (child == nullptr) {
        GNodeP nn = g_expand_child(g, root, a, planes, produced, path);
        if (!nn) return false;
        val = -nn->q;
    } else if (child->terminal) {
        path.push_back({root, a});
        val = -child->q;
    } else {
        path.push_back({root, a});
        double sub = 0.0;
        if (!g_sim_node(g, child, &sub, planes, produced, path)) return false;
        val = -sub;
    }
    root->N[(size_t)a] += 1;
    root->QSUM[(size_t)a] += (float)val;
    return true;
}

// pending 回填后的备份：沿路径自底向上交替取负（与 _simulate 的每层取负一致）
void g_backup_path(const std::vector<std::pair<GNode*, int>>& path, double leaf_q) {
    double v = leaf_q;
    for (int i = (int)path.size() - 1; i >= 0; --i) {
        GNode* nd = path[(size_t)i].first;
        int e = path[(size_t)i].second;
        const double rec = -v;
        nd->N[e] += 1;
        nd->QSUM[e] += (float)rec;
        v = rec;
    }
}

int g_n_rounds(int m) {
    if (m >= 2) {
        int r = 1;
        while ((1 << r) < m) r += 1;
        return r;
    }
    return 1;
}

void g_set_ks(GCtx& g, int budget) {
    int k = (int)g.cand.size();
    int base = budget / k, rem = budget % k;
    g.ks.assign((size_t)k, base);
    for (int i = 0; i < rem; ++i) g.ks[(size_t)i] += 1;
}

}  // namespace

KP_API void* kg_ctx_new(double c_visit, double c_scale, int m0, int claim_draw) {
    KP_TRY
    GCtx* g = new GCtx();
    g->c_visit = c_visit;
    g->c_scale = c_scale;
    g->m0 = m0;
    g->claim_draw = claim_draw != 0;
    return g;
    KP_CATCH(nullptr)
}

KP_API void kg_ctx_free(void* ctx) { delete (GCtx*)ctx; }

// 与 kp_set_root 同口径；返回根局面的合法着法数
KP_API int kg_set_root(void* ctx, const uint64_t* bbs, int turn, uint64_t castling, int ep,
                       int hmc, const uint16_t* moves, int n_moves) {
    KP_TRY
    GCtx& g = *(GCtx*)ctx;
    Pos p;
    std::memset(&p, 0, sizeof(p));
    for (int t = PAWN; t <= KING; ++t) p.bb[t] = bbs[t - 1];
    p.co[WHITE] = bbs[6];
    p.co[BLACK] = bbs[7];
    p.turn = turn ? WHITE : BLACK;
    p.castling = clean_castling(p, castling);
    p.ep = ep;
    p.hmc = hmc;
    g.root_hist.clear();
    std::vector<Mv> legal;
    for (int i = 0; i < n_moves; ++i) {
        Mv m = mv_decode(moves[i]);
        gen_legal(p, legal);
        bool ok = false;
        for (const Mv& x : legal)
            if (x.from == m.from && x.to == m.to && x.promo == m.promo) ok = true;
        if (!ok) {
            g_err = "对局历史第 " + std::to_string(i) + " 步不是合法着法";
            return -2;
        }
        g.root_hist.push_back(Hist{key_of(p), irreversible(p, m)});
        p = push(p, m);
    }
    g.root_pos = p;
    gen_legal(g.root_pos, legal);
    return (int)legal.size();
    KP_CATCH(-2)
}

// 根展开（前向已在 Python 侧做完）：返回合法着法数
KP_API int kg_expand_root(void* ctx, const float* policy, const float* promo, const float* wdl) {
    KP_TRY
    GCtx& g = *(GCtx*)ctx;
    std::vector<Mv> legal;
    gen_legal(g.root_pos, legal);
    GNodeP r = std::make_shared<GNode>();
    r->pos = g.root_pos;
    r->hist = g.root_hist;
    r->depth = 0;
    r->q = (double)wdl[0] - (double)wdl[2];
    std::vector<float> pri;
    priors_of(g.root_pos, legal, policy, promo, pri);
    g_fill_expanded(r.get(), legal, pri);
    g.root = r;
    return (int)legal.size();
    KP_CATCH(-2)
}

// noise 由 Python 采样（长度 = 根着法数）；g 已在 Python 侧乘进 noise
KP_API int kg_begin(void* ctx, const float* noise, int n_sims) {
    KP_TRY
    GCtx& g = *(GCtx*)ctx;
    // 同一个 ctx 可连续多次搜索（Player 每步一次）：这里必须整体复位，
    // 否则上一次的 finished/action/sims_used 会被当成这一次的结果返回。
    g.begun = true;
    g.finished = false;
    g.collecting = false;
    g.pending.clear();
    g.sims_used = 0;
    g.round = 0;
    g.wave = 0;
    g.ks.clear();
    g.cand.clear();
    g.cand_noise.clear();
    g.rounds_budget.clear();
    g.n_nodes = 0;
    g.n_terminal = 0;
    g.max_depth = 0;
    g.action = -1;
    const int n = (int)g.root->moves.size();
    std::vector<Mv> legal;
    gen_legal(g.root_pos, legal);
    double tv = 0.0;
    if (n == 0 || rules_value(g.root_pos, g.root_hist, g.root_hist.size(), legal,
                              g.claim_draw, &tv)) {
        g.begun = true;
        g.finished = true;
        g.action = -1;
        return 0;
    }
    std::vector<int> top((size_t)std::min(g.m0, n));
    kg_topm_impl(g.root->logits.data(), n, noise, g.m0, top.data());
    g.cand.assign(top.begin(), top.end());
    g.cand_noise.resize(g.cand.size());
    for (size_t i = 0; i < g.cand.size(); ++i) g.cand_noise[i] = noise[g.cand[i]];
    const int rounds = g_n_rounds((int)g.cand.size());
    int base = n_sims / rounds, rem = n_sims % rounds;
    g.rounds_budget.assign((size_t)rounds, base);
    for (int i = 0; i < rem; ++i) g.rounds_budget[(size_t)i] += 1;
    g.round = 0;
    g.wave = 0;
    g.sims_used = 0;
    {
        int budget = g.rounds_budget[0];
        if (g.cand.size() == 1) {
            budget = 0;
            for (int r2 = 0; r2 < rounds; ++r2) budget += g.rounds_budget[(size_t)r2];
        }
        g_set_ks(g, budget);
    }
    g.begun = true;
    return rounds;
    KP_CATCH(-2)
}

// 收集本拍需要前向的叶子；返回叶数（0 = 搜索完成）。info: [n_nodes,n_terminal,max_depth,
// sims_used,rounds,action]
KP_API int kg_collect(void* ctx, int cap, float* planes, int* info) {
    KP_TRY
    GCtx& g = *(GCtx*)ctx;
    if (!g.begun) throw std::runtime_error("kg_begin 尚未调用");
    if (g.collecting) throw std::runtime_error("上一拍尚未 kg_apply");
    auto fill = [&]() {
        info[0] = g.n_nodes;
        info[1] = g.n_terminal;
        info[2] = g.max_depth;
        info[3] = g.sims_used;
        info[4] = (int)g.rounds_budget.size();
        info[5] = g.action;
    };
    if (g.finished) {
        fill();
        return 0;
    }
    for (;;) {
        int maxk = 0;
        for (int k : g.ks) maxk = std::max(maxk, k);
        while (g.wave < maxk) {
            const int j = g.wave;
            int need = 0;
            for (size_t i = 0; i < g.cand.size(); ++i)
                if (j < g.ks[i]) need += 1;
            if (need > cap) throw std::runtime_error("kg_collect: cap 太小");
            int produced = 0;
            for (size_t i = 0; i < g.cand.size(); ++i) {
                if (j >= g.ks[i]) continue;
                g_sim_root(g, g.cand[i], planes, produced);   // 完成则已记录；pending 推迟到 apply
            }
            g.wave += 1;
            if (produced > 0) {
                g.collecting = true;
                fill();
                return produced;
            }
        }
        // 轮末：重打分（候选 = (g·noise) + ℓ + σ(completedQ)，float64 相加后稳定降序取半）
        {
            const int n = (int)g.root->moves.size();
            std::vector<float> pi((size_t)n), s((size_t)n);
            kg_softmax_into(g.root->logits.data(), n, pi.data());
            kg_qtransform(pi.data(), g.root->N.data(), g.root->QSUM.data(), n, g.root->q,
                          g.c_visit, g.c_scale, s.data());
            std::vector<std::pair<double, int>> scored;
            scored.reserve(g.cand.size());
            for (size_t i = 0; i < g.cand.size(); ++i) {
                const int a = g.cand[i];
                const double sc = (double)g.cand_noise[i] + (double)g.root->logits[(size_t)a]
                                  + (double)s[(size_t)a];
                scored.push_back({sc, (int)i});
            }
            std::stable_sort(scored.begin(), scored.end(),
                             [](const std::pair<double, int>& x, const std::pair<double, int>& y) {
                                 return x.first > y.first;
                             });
            const size_t keep = (size_t)std::max<size_t>(1, (g.cand.size() + 1) / 2);
            std::vector<int> nc;
            std::vector<float> nn2;
            for (size_t i = 0; i < keep; ++i) {
                nc.push_back(g.cand[(size_t)scored[i].second]);
                nn2.push_back(g.cand_noise[(size_t)scored[i].second]);
            }
            g.cand.swap(nc);
            g.cand_noise.swap(nn2);
        }
        g.sims_used += [&]() { int s2 = 0; for (int k : g.ks) s2 += k; return s2; }();
        g.round += 1;
        if (g.round >= (int)g.rounds_budget.size()) {
            g.finished = true;
            g.action = g.cand[0];
            fill();
            return 0;
        }
        int budget = g.rounds_budget[(size_t)g.round];
        if (g.cand.size() == 1) {
            budget = 0;
            for (int r2 = g.round; r2 < (int)g.rounds_budget.size(); ++r2)
                budget += g.rounds_budget[(size_t)r2];
        }
        g_set_ks(g, budget);
        g.wave = 0;
    }
    KP_CATCH(-2)
}

// 回填本拍的前向结果：policy/promo 是概率（与 kp_apply 同口径），n = 叶数
KP_API int kg_apply(void* ctx, const float* policy, const float* promo, const float* wdl, int n) {
    KP_TRY
    GCtx& g = *(GCtx*)ctx;
    if (!g.collecting) throw std::runtime_error("没有待回填的收集");
    if (n != (int)g.pending.size())
        throw std::runtime_error("kg_apply: 批大小与待回填叶子数不符");
    for (int i = 0; i < n; ++i) {
        Pending& p = g.pending[(size_t)i];
        std::vector<Mv> legal;
        for (uint16_t c : p.node->moves) legal.push_back(mv_decode(c));
        std::vector<float> pri;
        priors_of(p.node->pos, legal, policy + (size_t)i * 4096,
                  promo + (size_t)i * 4, pri);
        g_fill_expanded(p.node.get(), legal, pri);
        p.node->q = (double)wdl[(size_t)i * 3] - (double)wdl[(size_t)i * 3 + 2];
        g.n_nodes += 1;
        if (p.node->depth > g.max_depth) g.max_depth = p.node->depth;
        g_backup_path(p.path, p.node->q);
    }
    g.pending.clear();
    g.collecting = false;
    return 0;
    KP_CATCH(-2)
}

// 根 π′（长度 = 根着法数）
KP_API int kg_root_pi_prime(void* ctx, float* out) {
    KP_TRY
    GCtx& g = *(GCtx*)ctx;
    const int n = (int)g.root->moves.size();
    if (n <= 0) throw std::runtime_error("kg_root_pi_prime: 根无着法");
    kg_pi_prime_impl(g.root->logits.data(), n, g.root->N.data(), g.root->QSUM.data(),
                     g.root->q, g.c_visit, g.c_scale, out);
    return 0;
    KP_CATCH(-2)
}

KP_API int kg_encode_root(void* ctx, float* out) {
    KP_TRY
    GCtx& g = *(GCtx*)ctx;
    encode(g.root_pos, repetitions_of(g.root_pos, g.root_hist, g.root_hist.size()), out);
    return 0;
    KP_CATCH(-2)
}

// 导出根的着法/ℓ/N/QSUM/q（Python 侧重建 Node 供 π′ 与测试用）；返回着法数
KP_API int kg_root_export(void* ctx, uint16_t* moves, float* logits, int64_t* N, float* QSUM,
                          double* q) {
    KP_TRY
    GCtx& g = *(GCtx*)ctx;
    const int n = (int)g.root->moves.size();
    for (int i = 0; i < n; ++i) {
        if (moves) moves[i] = g.root->moves[(size_t)i];
        if (logits) logits[i] = g.root->logits[(size_t)i];
        if (N) N[i] = g.root->N[(size_t)i];
        if (QSUM) QSUM[i] = g.root->QSUM[(size_t)i];
    }
    if (q) *q = g.root->q;
    return n;
    KP_CATCH(-2)
}
