#!/usr/bin/env python3
"""
core_pass_oracle_final.py

Квантовый (симулированный) генератор паролей + оракул безопасности с проверкой Левенштейна.
- QuantumSimulator: 6-кубитный симулятор (numpy)
- PasswordGenerator: буквы + квантовые цифры + символы (secrets)
- PasswordOracle: проверка требований + Levenshtein (threshold=5) + усиление
- Режимы длины: 12 / 16 / 18
"""

from __future__ import annotations

import secrets
import string
from typing import List, Optional, Set, Tuple

import numpy as np

# -----------------------------
# Конфигурация
# -----------------------------
SYMBOLS = "!@#$%^&*()-_=+[]{};:,.<>/?"
RECOMMENDED_LENGTHS = {1: 12, 2: 16, 3: 18}
DEFAULT_MODE = 1
LEVENSHTEIN_THRESHOLD = 5

COMMON_PASSWORDS: Set[str] = {
    "123456", "12345678", "123456789", "password", "qwerty", "abc123",
    "111111", "123123", "admin", "letmein", "welcome", "iloveyou",
    "monkey", "dragon", "sunshine", "princess", "football",
    # локальные примеры
    "пароль", "йцукен", "пароль123"
}

# -----------------------------
# QuantumSimulator (6 кубитов)
# -----------------------------


class QuantumSimulator:
    """
    Симулятор 6-кубитного квантового генератора цифр 0..9.
    Реализация: готовим состояние |0...0>, применяем Hadamard ко всем кубитам,
    вычисляем вероятности амплитуд, делаем выбор базисного состояния (коллапс),
    переводим индекс в 6 бит и возвращаем число % 10 при условии index < 60.
    """

    def __init__(self, n_qubits: int = 6):
        self.n_qubits = int(n_qubits)
        # локально не храним состояние между вызовами, каждый вызов заново инициализирует
        # но оставим метод для совместимости
        self.state = self._zero_state()

    def _zero_state(self) -> np.ndarray:
        vec = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        vec[0] = 1.0
        return vec

    def _hadamard_all(self) -> np.ndarray:
        """Возвращает состояние после применения Hadamard ко всем кубитам."""
        # Hadamard на один кубит
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        # Построим оператор H^{\otimes n} как тензорное произведение
        op = H
        for _ in range(self.n_qubits - 1):
            op = np.kron(op, H)
        # применим к |0...0> (оптимально — умножение матрицы большого размера; для n=6 OK)
        zero = self._zero_state()
        return op @ zero

    def generate_random_digit(self) -> int:
        """
        Генерирует цифру 0..9 равномерно:
          - генерируем состояние (Hadamard на всех кубитах)
          - берём вероятности |amp|^2
          - выбираем базисный индекс с помощью secrets-рандома
          - если index < 60 -> return index % 10, иначе повторяем
        """
        while True:
            state = self._hadamard_all()
            probs = np.abs(state) ** 2
            # нормировка (на всякий случай)
            probs = probs / probs.sum()

            # cumulate
            cum = np.cumsum(probs)

            # secrets.randbelow для выбора: получим случайное число r в [0,1)
            r = secrets.randbelow(10**9) / 10**9

            # найти индекс такого, что cum[index] > r
            idx = int(np.searchsorted(cum, r, side="right"))
            if idx < 60:  # допускаем индексы 0..59
                return int(idx % 10)
            # иначе повторяем цикл


# -----------------------------
# Levenshtein distance
# -----------------------------
def levenshtein(a: str, b: str) -> int:
    """Вычисление расстояния Левенштейна — классическое DP (итеративно)."""
    a = a or ""
    b = b or ""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # используем только две строки
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]

# -----------------------------
# PasswordGenerator
# -----------------------------


class PasswordGenerator:
    """Генератор паролей: буквы (upper+lower), квантовые цифры, символы (secrets)."""

    def __init__(self, quantum_sim: Optional[QuantumSimulator] = None):
        self.qsim = quantum_sim or QuantumSimulator()
        self.letters = string.ascii_letters
        self.symbols = SYMBOLS

    def generate(self, length: int) -> str:
        parts: List[str] = []
        for _ in range(length):
            t = secrets.randbelow(3)  # 0 = letter, 1 = digit, 2 = symbol
            if t == 0:
                parts.append(secrets.choice(self.letters))
            elif t == 1:
                parts.append(str(self.qsim.generate_random_digit()))
            else:
                parts.append(secrets.choice(self.symbols))
        return "".join(parts)

# -----------------------------
# PasswordOracle
# -----------------------------


class PasswordOracle:
    """
    Проверка пароля:
      - минимальная длина
      - наличие upper/lower/digit/symbol
      - проверка по common (точное совпадение)
      - проверка по Levenshtein (threshold = 5)
    И автокоррекция (strengthen), использующая добавление/замены символов с secrets
    """

    def __init__(self,
                 common: Optional[Set[str]] = None,
                 lev_threshold: int = LEVENSHTEIN_THRESHOLD):
        self.common = common or COMMON_PASSWORDS
        self.lev_threshold = int(lev_threshold)
        self.symbols = SYMBOLS

    def check_basic(self, pwd: str, min_length: int) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if len(pwd) < min_length:
            issues.append(f"слишком короткий (нужно >= {min_length})")
        if not any(c.isupper() for c in pwd):
            issues.append("нет заглавной буквы")
        if not any(c.islower() for c in pwd):
            issues.append("нет строчной буквы")
        if not any(c.isdigit() for c in pwd):
            issues.append("нет цифры")
        if not any(c in self.symbols for c in pwd):
            issues.append("нет специального символа")
        if pwd.lower() in self.common:
            issues.append("очевидный/common пароль")
        return (len(issues) == 0, issues)

    def check_levenshtein(self, pwd: str) -> Tuple[bool, Optional[str], int]:
        """Возвращает (is_far_enough, closest_common_or_none, min_distance)."""
        pw = pwd.lower()
        min_d = None
        closest = None
        for c in self.common:
            d = levenshtein(pw, c.lower())
            if min_d is None or d < min_d:
                min_d = d
                closest = c
                if min_d == 0:
                    break
        if min_d is None:
            return True, None, 999
        return (min_d > self.lev_threshold, closest, min_d)

    def check_password(self, pwd: str, min_length: int) -> Tuple[bool, List[str]]:
        ok_basic, issues = self.check_basic(pwd, min_length)
        ok_lev, closest, dist = self.check_levenshtein(pwd)
        if not ok_lev:
            issues.append(
                f"слишком похож на '{closest}' (lev={dist} ≤ {self.lev_threshold})")
        return (ok_basic and ok_lev, issues)

    def strengthen(self,
                   pwd: str,
                   min_length: int,
                   qsim: Optional[QuantumSimulator] = None,
                   recursion_limit: int = 4,
                   depth: int = 0) -> str:
        """
        Усиление:
          - добавляет/заменяет символы чтобы удовлетворить базовым требованиям
          - если слишком похож на common — делает агрессивные замены/вставки
          - использует qsim для цифр
          - защищено от бесконечной рекурсии
        """
        if depth > recursion_limit:
            return pwd
        qsim = qsim or QuantumSimulator()
        arr = list(pwd)

        # если точное совпадение с common
        if "".join(arr).lower() in self.common:
            arr.append(secrets.choice(self.symbols))
            arr.append(str(qsim.generate_random_digit()))

        # гарантируем верхний/нижний/цифру/символ
        if not any(c.isupper() for c in arr):
            # попытка заменить первую строчную
            for i, ch in enumerate(arr):
                if ch.islower():
                    arr[i] = ch.upper()
                    break
            else:
                arr.insert(0, secrets.choice(string.ascii_uppercase))

        if not any(c.islower() for c in arr):
            for i, ch in enumerate(arr):
                if ch.isupper():
                    arr[i] = ch.lower()
                    break
            else:
                arr.append(secrets.choice(string.ascii_lowercase))

        if not any(c.isdigit() for c in arr):
            replaced = False
            for i, ch in enumerate(arr):
                if not ch.isdigit():
                    arr[i] = str(qsim.generate_random_digit())
                    replaced = True
                    break
            if not replaced:
                arr.append(str(qsim.generate_random_digit()))

        if not any(c in self.symbols for c in arr):
            replaced = False
            for i, ch in enumerate(arr):
                if ch.isalnum():
                    arr[i] = secrets.choice(self.symbols)
                    replaced = True
                    break
            if not replaced:
                arr.append(secrets.choice(self.symbols))

        # удлиняем до min_length
        while len(arr) < min_length:
            t = secrets.randbelow(3)
            if t == 0:
                arr.append(secrets.choice(string.ascii_letters))
            elif t == 1:
                arr.append(str(qsim.generate_random_digit()))
            else:
                arr.append(secrets.choice(self.symbols))

        # если слишком похож на common — агрессивно заменим несколько символов
        ok_lev, closest, dist = self.check_levenshtein("".join(arr))
        if not ok_lev:
            length = len(arr)
            swaps = min(3, max(1, length // 6))
            for _ in range(swaps):
                idx = secrets.randbelow(length)
                kind = secrets.randbelow(3)
                if kind == 0:
                    arr[idx] = secrets.choice(string.ascii_letters)
                elif kind == 1:
                    arr[idx] = str(qsim.generate_random_digit())
                else:
                    arr[idx] = secrets.choice(self.symbols)
            # вставим случайный символ в случайное место
            pos = secrets.randbelow(len(arr) + 1)
            arr.insert(pos, secrets.choice(self.symbols))

        # лёгкая перестановка
        if len(arr) >= 4:
            i = secrets.randbelow(len(arr))
            j = secrets.randbelow(len(arr))
            arr[i], arr[j] = arr[j], arr[i]

        candidate = "".join(arr)

        ok, issues = self.check_password(candidate, min_length)
        if ok:
            return candidate

        # если ещё остаются проблемы — добавляем недостающие элементы и проверяем расстояние
        if any("нет заглавной" in it for it in issues) or any("нет строчной" in it for it in issues):
            if not any(c.isupper() for c in candidate):
                candidate += secrets.choice(string.ascii_uppercase)
            if not any(c.islower() for c in candidate):
                candidate += secrets.choice(string.ascii_lowercase)
        if any("нет цифры" in it for it in issues):
            candidate += str(qsim.generate_random_digit())
        if any("нет специального символа" in it for it in issues):
            candidate += secrets.choice(self.symbols)
        if any("слишком корот" in it for it in issues):
            while len(candidate) < min_length:
                candidate += secrets.choice(string.ascii_letters +
                                            self.symbols + string.digits)

        # снова проверить расстояние Левенштейна
        ok_lev2, closest2, dist2 = self.check_levenshtein(candidate)
        if not ok_lev2 and depth < recursion_limit:
            candidate = self.strengthen(
                candidate, min_length, qsim=qsim, recursion_limit=recursion_limit, depth=depth + 1)

        return candidate

# -----------------------------
# CLI / Main
# -----------------------------


def choose_mode() -> int:
    print("Выберите режим длины пароля:")
    print(" 1 — 12 символов (стандарт)")
    print(" 2 — 16 символов (сильный)")
    print(" 3 — 18 символов (параноидальный)")
    while True:
        choice = input(f"Режим (Enter для {DEFAULT_MODE}): ").strip()
        if choice == "":
            return DEFAULT_MODE
        if choice in {"1", "2", "3"}:
            return int(choice)
        print("Некорректный ввод. Введите 1, 2 или 3.")


def ask_int(prompt: str, default: Optional[int] = None, minimum: Optional[int] = None) -> int:
    raw = input(prompt).strip()
    if raw == "":
        if default is not None:
            return default
        raise ValueError("Ожидалось число, получено пустая строка")
    val = int(raw)
    if minimum is not None and val < minimum:
        raise ValueError(f"Значение должно быть не менее {minimum}")
    return val


def main() -> None:
    print("[LOCK] Quantum Password Generator + Oracle (Levenshtein threshold = 5)")
    print("-" * 72)

    mode = choose_mode()
    length = RECOMMENDED_LENGTHS[mode]
    try:
        count = ask_int(
            "Сколько паролей сгенерировать (Enter для 1): ", default=1, minimum=1)
    except ValueError as e:
        print(f"[ERR] {e}")
        return

    auto_fix_choice = input(
        "Автоисправление слабых/похожих паролей? (Y/n): ").strip().lower()
    auto_fix = (auto_fix_choice != "n")

    qsim = QuantumSimulator()
    gen = PasswordGenerator(qsim)
    oracle = PasswordOracle(common=COMMON_PASSWORDS,
                            lev_threshold=LEVENSHTEIN_THRESHOLD)

    for i in range(count):
        raw = gen.generate(length)
        print(f"\n{i + 1}. Исходный: {raw}")

        ok, issues = oracle.check_password(raw, length)
        if ok:
            print("    [OK] Пароль соответствует требованиям.")
        else:
            print(f"    [WARN] Проблемы: {', '.join(issues)}")
            if auto_fix:
                fixed = oracle.strengthen(raw, length, qsim=qsim)
                ok2, issues2 = oracle.check_password(fixed, length)
                print(f"    [FIXED] {fixed}")
                if not ok2:
                    print(
                        f"    [WARN] Исправленный пароль всё ещё имеет проблемы: {issues2}")
            else:
                print("    (Автоисправление отключено)")

    print("\nГотово.")


if __name__ == "__main__":
    main()
