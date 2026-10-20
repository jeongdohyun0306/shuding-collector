import re
from functools import lru_cache

NAMES = ["슈딩이"]
CAT = "고양이"

NOUN_TAGS = {"NNG", "NNP", "NNB", "NP", "NR", "SN", "SL", "SH", "XSN", "XPN"}
PRED_TAGS = {"VV", "VA", "VX", "XSV", "XSA"}
ENDING_TAGS = {"ETM", "EC", "EF", "ETN"}
AUX_LEMMAS = {"있", "보", "싶", "주", "버리", "놓", "두", "계시", "가", "오"}
LOC_VERBS = {"나가", "들어가", "올라가", "내려가", "들어오", "올라오", "내려오", "나오", "눕", "앉", "있", "매달리",
             "숨", "서", "안기", "올라타", "기대", "엎드리", "붙", "끼"}
CAT_HEADS = {CAT, "사진", "모습", "장면", "것", "거", "때", "얼굴", "표정", "컷", "짤", "순간"}
BEING_NOUNS = {"사람", "학생", "선생", "아이", "어린이", "친구", "주인", "강아지", "개", "새", "누구", "아무", "엄마",
               "아빠", "형", "누나", "언니", "오빠", "동생", "애", "남자", "여자", "어른", "손", "고양이", "냥이", "동물",
               "선배", "후배", "쌤", "집사", "사람들", "애들"}
PLACE_NOUNS = {"침대", "소파", "쇼파", "의자", "실내", "밖", "집", "교실", "복도", "상자", "박스", "가방", "창가", "창문",
               "풀밭", "잔디", "잔디밭", "차", "자동차", "바닥", "책상", "이불", "담요", "계단", "나무", "화단", "운동장", "방",
               "안", "위", "밑", "옆", "속", "아래", "앞", "뒤", "베란다", "현관", "화장실", "세면대", "부엌", "식탁", "벤치",
               "길", "도로", "주차장", "기숙사", "급식실", "도서관", "강당", "옥상", "정원", "숲", "흙", "모래"}
ACCESSORY = {"모자", "안경", "선글라스", "마스크", "헬멧", "왕관", "고깔", "옷", "셔츠", "티셔츠", "스웨터", "원피스", "한복",
             "망토", "잠옷", "코트", "패딩", "후드", "조끼", "신발", "양말", "장화", "목줄", "하네스", "방울", "목걸이",
             "넥타이", "리본", "나비넥타이", "목도리", "스카프", "장갑", "반지", "시계", "머리띠", "두건", "턱받이", "넥카라",
             "카라", "깔때기", "가운", "천", "니트", "점퍼", "바지", "치마", "케이프"}
WEAR_VERBS = {"쓰", "입", "끼", "신", "차", "매", "두르", "하", "착용하", "걸치", "달"}
IDIOMS = ("어이없", "어처구니없", "힘없", "정신없", "재미없", "상관없", "끝없", "틀림없", "쉴새없", "할일없", "별수없",
          "문제없", "소용없", "관계없", "거침없", "쓸데없", "다름없", "맛없", "멋없", "버릇없", "철없", "하염없", "뜬금없",
          "사정없", "가차없", "빈틈없", "아낌없", "어림없", "두말없", "난데없", "간데없", "꼼짝없", "영락없", "유례없",
          "염치없", "부질없", "속절없", "터무니없", "실없", "열없", "물샐틈없", "기운없", "생각없", "개념없", "의미없",
          "필요없", "변함없", "끊임없", "한없", "말없", "소리없", "흠없", "티없", "어김없", "여지없", "정처없")
MOT_LEXICAL = {"생기", "되", "나", "마땅하"}
HONORIFIC = {"계시": "있", "주무시": "자", "드시": "먹", "잡수시": "먹"}
EXCEPTIVE = {"말", "빼", "제외하"}
COMITATIVE = {"랑", "이랑", "와", "과", "하고", "한테", "에게", "께", "한테서", "에게서"}
EXISTENTIAL = {"있", "계시"}
GENERIC = {"있는", "하는", "되는", "있는데", "함께", "같이", "는", "은", CAT}
POSITIONAL = {"안", "위", "밑", "옆", "속", "아래", "앞", "뒤", "근처", "사이", "쪽"}
BOUND_PLACES = {"데", "곳", "쪽", "것", "거"}
FRAME_NOUNS = {"사진", "모습", "장면", "컷", "짤", "순간", "때"}
OTHER_MM = {"다른", "딴"}
COORD_JX = {"이나", "나", "이든", "든", "이든지", "든지"}
VISIBLE = {"보이", "나오", "찍히"}


@lru_cache(maxsize=1)
def kiwi():
    from kiwipiepy import Kiwi
    k = Kiwi()
    k.add_user_word(CAT, "NNG")
    return k


def base(tag):
    return tag.split("-")[0]


def has_batchim(s):
    ch = s[-1]
    code = ord(ch) - 0xAC00
    return 0 <= code < 11172 and code % 28 != 0


def josa(noun, with_b, without_b):
    return noun + (with_b if has_batchim(noun) else without_b)


def name_pattern(names):
    alts = []
    for n in names:
        if n.endswith("이") and len(n) > 2:
            alts.append(re.escape(n[:-1]) + "(?:이|아)?")
        else:
            alts.append(re.escape(n))
    return re.compile("|".join(alts))


def normalize(q, names):
    q = name_pattern(names).sub(CAT, q)
    q = re.sub(r"(?<![가-힣])냥이", CAT, q)
    return re.sub(r"\s+", " ", q).strip()


class Parse:
    def __init__(self, q):
        self.q = q
        self.t = kiwi().tokenize(q)
        self.drop = set()
        self.neg = []

    def tag(self, i):
        return base(self.t[i].tag) if 0 <= i < len(self.t) else ""

    def form(self, i):
        return self.t[i].form if 0 <= i < len(self.t) else ""

    def span(self, i, j):
        return self.q[self.t[i].start:self.t[j].end].strip()

    def mark(self, i, j):
        for k in range(i, j + 1):
            self.drop.update(range(self.t[k].start, self.t[k].end))

    def pred_start(self, j):
        while j - 1 >= 0 and self.tag(j) in ("XSV", "XSA") and self.tag(j - 1) in NOUN_TAGS:
            j -= 1
        return j

    def unit_end(self, j):
        k = j
        while True:
            nt = self.tag(k + 1)
            if nt == "EP":
                k += 1
            elif nt in ("XSV", "XSA"):
                k += 1
            elif nt == "EC" and self.tag(k + 2) == "VX" and self.form(k + 2) in AUX_LEMMAS:
                k += 2
            elif nt == "EC" and self.form(k + 1) == "게" and self.tag(k + 2) in ("VV", "VA"):
                k += 2
            elif nt in ENDING_TAGS:
                return k + 1
            else:
                return k

    def head_noun(self, k):
        for m in range(k + 1, min(k + 4, len(self.t))):
            if self.tag(m) in NOUN_TAGS:
                return self.form(m)
            if self.tag(m) not in ("MAG", "MM", "SP", "SS"):
                return None
        return None

    def adnominal(self, i, j, past=None):
        morphs = []
        for k in range(i, j + 1):
            f, tg = self.t[k].form, self.t[k].tag
            b = base(tg)
            if b == "EP" and f in ("시", "으시"):
                continue
            if b in ("VV", "VX", "VA") and f in HONORIFIC:
                f = HONORIFIC[f]
            morphs.append((f, tg))
        while morphs and base(morphs[-1][1]) in ("EC", "EF", "ETN"):
            morphs.pop()
        if not morphs:
            return self.span(i, j)
        if base(morphs[-1][1]) != "ETM":
            last = morphs[-1][0]
            if last in ("있", "없", "계시"):
                morphs.append(("는", "ETM"))
            else:
                morphs.append(("ᆫ", "ETM") if base(morphs[-1][1]) == "VA" or past else ("는", "ETM"))
        try:
            return kiwi().join(morphs)
        except Exception:
            return self.span(i, j)

    def noun_chain_left(self, k):
        start = k
        while start - 1 >= 0 and (self.tag(start - 1) in NOUN_TAGS or self.tag(start - 1) == "MM"
                                  or (self.tag(start - 1) == "JKG" and self.tag(start - 2) in NOUN_TAGS)):
            if self.form(start - 1) in POSITIONAL and self.tag(start - 1) in NOUN_TAGS:
                break
            start -= 1
        if self.tag(start - 1) == "ETM" and self.tag(start - 2) in PRED_TAGS:
            if self.form(start) in BOUND_PLACES or self.form(start) in FRAME_NOUNS:
                ps = self.pred_start(start - 2)
                a = self.args_left(ps, self.form(ps))
                start = a[0][0] if a else ps
            elif self.tag(start - 2) == "VA":
                start = start - 2
        return start

    def head_of(self, s, e):
        for k in range(e, s - 1, -1):
            if self.tag(k) in ("NNG", "NNP", "NP", "NNB"):
                return self.form(k)
        return self.form(e)

    def self_cat(self, s, e):
        if CAT not in self.span(s, e):
            return False
        return not (self.tag(s) == "MM" and self.form(s) in OTHER_MM)

    def coord_at(self, k):
        return self.tag(k) == "JC" or (self.tag(k) == "JX" and self.form(k) in COORD_JX)

    def noun_kind(self, s, e):
        h = self.head_of(s, e)
        for name, group in (("being", BEING_NOUNS), ("wear", ACCESSORY), ("place", PLACE_NOUNS)):
            if h in group:
                return name
        return None

    def args_left(self, i, verb):
        args = []
        k = i - 1
        while k >= 0 and self.tag(k) == "MAG" and self.form(k) not in ("안", "못"):
            args.insert(0, (k, k, None))
            k -= 1
        used = set()
        while k >= 0:
            p = None
            if self.tag(k) in ("JKO", "JKS", "JX", "JKB"):
                p = k
                k -= 1
            if self.tag(k) not in NOUN_TAGS:
                break
            s = self.noun_chain_left(k)
            if self.self_cat(s, k):
                break
            while self.coord_at(s - 1) and self.tag(s - 2) in NOUN_TAGS:
                s2 = self.noun_chain_left(s - 2)
                if self.self_cat(s2, s - 2):
                    break
                s = s2
            ptag = self.tag(p) if p is not None else "BARE"
            if ptag == "JKB" and self.form(p) in COMITATIVE:
                ptag = "COM"
            if ptag == "JKB" and verb not in LOC_VERBS:
                break
            if ptag in used:
                break
            used.add(ptag)
            args.insert(0, (s, p if p is not None else k, ptag))
            k = s - 1
        return args

    def object_noun(self, args):
        for s, e, ptag in reversed(args):
            if ptag in ("JKO", "BARE", "JX"):
                end = e - 1 if ptag != "BARE" else e
                return self.form(end), self.span(s, end)
        return None, None

    def predicate_phrase(self, args, ps, ue, past=None):
        lemma = self.form(ps) if self.tag(ps) not in NOUN_TAGS else self.form(ps + 1)
        noun, noun_text = self.object_noun(args)
        if noun == "아무것" and lemma in WEAR_VERBS:
            noun, noun_text = "옷", "옷"
        if noun and noun in ACCESSORY and lemma in WEAR_VERBS:
            return f"{josa(noun_text, '을', '를')} 착용한 {CAT}"
        if noun and lemma == "하" and self.tag(ps) not in NOUN_TAGS and noun not in ACCESSORY:
            others = [self.span(s, e) for s, e, p in args if p not in ("JKO", "BARE", "JX")]
            return " ".join(others + [noun_text + self.adnominal(ps, ue, past), CAT])
        parts = [self.span(s, e) for s, e, _ in args] + [self.adnominal(ps, ue, past), CAT]
        return " ".join(parts)

    def short_neg(self, i):
        j = i + 1
        if self.tag(j) not in ("VV", "VA", "VX") and not (self.tag(j) in NOUN_TAGS and self.tag(j + 1) in ("XSV", "XSA")):
            return False
        lemma = self.form(j)
        if self.form(i) == "못" and lemma in MOT_LEXICAL:
            return False
        ue = self.unit_end(j)
        if lemma in EXISTENTIAL or lemma in VISIBLE:
            subj = [(s_, e_) for s_, e_, pt in self.args_left(i, lemma) if pt in ("JKS", "JX", "BARE")]
            if subj:
                s_, e_ = subj[-1]
                end = e_ - 1 if self.tag(e_) in ("JKS", "JX") else e_
                if lemma in EXISTENTIAL or self.noun_kind(s_, end) == "being":
                    self.neg.append(f"{josa(self.span(s_, end), '과', '와')} 함께 있는 {CAT}")
                    self.mark(subj[0][0], ue)
                    return True
        if self.tag(ue) == "ETM":
            h = self.head_noun(ue)
            if h is not None and h not in CAT_HEADS:
                return False
        args = self.args_left(i, lemma)
        self.neg.append(self.predicate_phrase(args, j, ue))
        self.mark(args[0][0] if args else i, ue)
        return True

    def long_neg(self, n):
        k = n - 1
        while k >= 0 and self.tag(k) == "JX":
            k -= 1
        if not (self.tag(k) == "EC" and self.form(k) == "지"):
            return False
        p = k - 1
        while p - 2 >= 0 and self.tag(p) == "VX" and self.tag(p - 1) == "EC":
            p -= 2
        while p - 1 >= 0 and self.tag(p) == "EP":
            p -= 1
        if self.tag(p) not in PRED_TAGS:
            return False
        ps = self.coord_pred(self.pred_start(p))
        ue = self.unit_end(n)
        if self.tag(ue) == "ETM":
            h = self.head_noun(ue)
            if h is not None and h not in CAT_HEADS:
                return False
        lemma = self.form(ps) if self.tag(ps) not in NOUN_TAGS else self.form(ps + 1)
        args = self.args_left(ps, lemma)
        past = any(self.tag(m) == "EP" and self.form(m) in ("었", "았", "였") for m in range(n, ue + 1)) or             (self.tag(ue) == "ETM" and self.form(ue) in ("ᆫ", "은", "던"))
        phrase = self.predicate_phrase(args, ps, k - 1, past)
        start = args[0][0] if args else ps
        first = start
        while self.tag(start - 1) == "EC" and self.form(start - 1) in ("거나", "나", "든지", "든가") and self.tag(start - 2) in PRED_TAGS:
            ps2 = self.pred_start(start - 2)
            a2 = self.args_left(ps2, self.form(ps2))
            start = a2[0][0] if a2 else ps2
        if start != first:
            phrase = self.q[self.t[start].start:self.t[first].start].strip() + " " + phrase
        self.neg.append(phrase)
        self.mark(start, ue)
        return True

    def coord_pred(self, ps):
        while self.tag(ps - 1) == "EC" and self.form(ps - 1) in ("거나", "나", "든지", "든가") and self.tag(ps - 2) in PRED_TAGS:
            ps = self.pred_start(ps - 2)
        return ps

    def np_left(self, k):
        if self.tag(k) in ("JKC", "JKS", "JX", "JKO"):
            k -= 1
        if self.tag(k) not in NOUN_TAGS:
            return None
        s = self.noun_chain_left(k)
        if self.form(k) in ("것", "거", "게") and s < k:
            return s, k, "clause"
        if self.form(k) in FRAME_NOUNS and any(self.tag(m) == "ETM" for m in range(s, k)):
            return s, k, "clause"
        return s, k, "np"

    def coord_left(self, s, e):
        groups = []
        kind = self.noun_kind(s, e)
        k = s - 1
        while k >= 0:
            comma = self.tag(k) == "SP" and self.form(k) == ","
            if self.coord_at(k) or (self.tag(k) == "JX" and self.form(k) == "도") or comma:
                if self.tag(k - 1) not in NOUN_TAGS:
                    break
                s2 = self.noun_chain_left(k - 1)
                if comma and (kind is None or self.noun_kind(s2, k - 1) != kind):
                    break
                groups.insert(0, (s2, k - 1))
                k = s2 - 1
            else:
                break
        return groups

    def template(self, s, e, kind, absence=False):
        text = self.span(s, e)
        head = self.head_of(s, e)
        if kind == "clause":
            return f"{re.sub(r'[ ]*(것|거|게|' + '|'.join(FRAME_NOUNS) + r')$', '', text)} {CAT}"
        if head == CAT and text != CAT and not absence:
            return text
        if head == "혼자":
            return f"혼자 있는 {CAT}"
        if head == "아무":
            return f"사람과 함께 있는 {CAT}"
        if head in BEING_NOUNS or (CAT in text and text != CAT):
            return f"{josa(text, '과', '와')} 함께 있는 {CAT}"
        if head in ACCESSORY:
            return f"{josa(text, '을', '를')} 착용한 {CAT}"
        if head in PLACE_NOUNS:
            return f"{text}에 있는 {CAT}"
        if absence:
            return f"{josa(text, '과', '와')} 함께 있는 {CAT}"
        return f"{text} {CAT}"

    def absence(self, i):
        q2 = self.q[: self.t[i].start].replace(" ", "")
        if any((q2 + "없").endswith(w) for w in IDIOMS):
            return False
        k = i - 1
        r = self.np_left(k)
        if r is None:
            return False
        s, e, kind = r
        groups = self.coord_left(s, e) + [(s, e)]
        if self.tag(s - 1) == "MM" and self.form(s - 1) == "다른":
            groups[-1] = (s - 1, e)
        first = groups[0][0]
        ue = self.unit_end(i) if self.tag(i) == "VA" else i
        for gs, ge in groups:
            self.neg.append(self.template(gs, ge, kind if (gs, ge) == (s, e) else "np", absence=True))
        self.mark(first, ue)
        return True

    def np_operator(self, i, end):
        r = self.np_left(i - 1)
        if r is None:
            return False
        s, e, kind = r
        groups = self.coord_left(s, e) + [(s, e)]
        if self.form(e) == CAT and s == e:
            return False
        for gs, ge in groups:
            self.neg.append(self.template(gs, ge, kind if (gs, ge) == (s, e) else "np"))
        self.mark(groups[0][0], end)
        return True

    def run(self):
        i = 0
        while i < len(self.t):
            f, tg = self.form(i), self.tag(i)
            done = False
            if tg == "MAG" and f in ("안", "못"):
                done = self.short_neg(i)
            elif tg in ("VX", "VV") and f in ("않", "못하", "말") and self.tag(i - 1) in ("EC", "JX"):
                done = self.long_neg(i)
            elif (tg == "VA" and f == "없") or (tg == "MAG" and f == "없이"):
                done = self.absence(i)
            elif tg == "VCN" and f == "아니":
                done = self.np_operator(i, self.unit_end(i))
            elif tg == "VV" and f in EXCEPTIVE and self.form(i + 1) == "고" and not (self.form(i - 1) == "지" and self.tag(i - 1) == "EC"):
                done = self.np_operator(i, i + 1)
            elif tg == "NNG" and f == "제외" and self.tag(i + 1) == "XSV" and self.form(i + 1) == "하" and self.form(i + 2) == "고":
                done = self.np_operator(i, i + 2)
            i += 1
        return self

    def pos(self):
        chars = [c for k, c in enumerate(self.q) if k not in self.drop]
        p = "".join(chars)
        p = re.sub(r"(지만|는데)(?=\s|$)", "는", p)
        p = re.sub(r"(인데)(?=\s|$)", "인", p)
        p = re.sub(r"\s+", " ", p).strip()
        if self.drop:
            words = []
            for w in p.split(" "):
                if re.fullmatch(r"(다른|딴)?(곳|데)(에서|에|의|서)?|거|것", w):
                    if words and words[-1] in OTHER_MM:
                        words.pop()
                    continue
                words.append(w)
            p = " ".join(words)
        if CAT not in p:
            p = (p + " " + CAT).strip()
        return p


def useful(phrase):
    words = [w for w in re.split(r"\s+", phrase) if w not in GENERIC]
    return bool(words)


def parse(q, names=NAMES):
    pr = Parse(normalize(q, names)).run()
    neg = []
    for n in pr.neg:
        n = re.sub(r"\s+", " ", n).strip()
        if useful(n) and n not in neg:
            neg.append(n)
    return {"pos": pr.pos(), "neg": neg}
