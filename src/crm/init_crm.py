"""
init_crm.py — Inicialización del CRM SQLite para el Agente de Cobranza Level 3.

Responsabilidades:
  1. Crear las 5 tablas si no existen (idempotente).
  2. Sembrar datos sintéticos realistas si la DB está vacía.
  3. Proveer get_db_path() como única fuente de verdad de la ruta del archivo.

Diseñado para ejecutarse en cada arranque de la app en Streamlit Cloud
(filesystem efímero). El .db va en .gitignore — nunca en el repositorio.

Uso en app.py / tab_cobranza.py:
    from src.crm.init_crm import inicializar_crm
    inicializar_crm()   # idem-potente, seguro llamarlo en cada arranque

Uso standalone para reset del demo:
    python src/crm/init_crm.py --reset
"""

import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "cobranza_crm.db"


def get_db_path() -> Path:
    """Única fuente de verdad de la ruta del CRM. Usar en todos los módulos."""
    return DB_PATH


# ---------------------------------------------------------------------------
# DDL — 5 tablas
# ---------------------------------------------------------------------------

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS clientes (
        cliente_id         TEXT    PRIMARY KEY,
        nombre             TEXT    NOT NULL,
        ews_score          REAL    NOT NULL,
        dias_mora          INTEGER NOT NULL DEFAULT 0,
        monto_adeudado     REAL    NOT NULL,
        limite_credito     REAL    NOT NULL,
        etapa_cobranza     TEXT    NOT NULL DEFAULT 'preventiva',
        gestor_asignado    TEXT    NOT NULL DEFAULT 'Agente_L3',
        gestionado_humano  INTEGER NOT NULL DEFAULT 0,
        ultimo_contacto    TEXT,
        fecha_alta         TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS contactos (
        contacto_id  TEXT PRIMARY KEY,
        cliente_id   TEXT NOT NULL REFERENCES clientes(cliente_id),
        canal        TEXT NOT NULL,
        resultado    TEXT NOT NULL,
        notas        TEXT NOT NULL DEFAULT '',
        agente       TEXT NOT NULL DEFAULT 'Agente_L3',
        timestamp    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS escalaciones (
        escalacion_id    TEXT PRIMARY KEY,
        cliente_id       TEXT NOT NULL REFERENCES clientes(cliente_id),
        nivel            TEXT NOT NULL,
        motivo           TEXT NOT NULL,
        estado           TEXT NOT NULL DEFAULT 'pendiente',
        resuelto_por     TEXT,
        fecha_resolucion TEXT,
        timestamp        TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS propuestas (
        propuesta_id     TEXT    PRIMARY KEY,
        cliente_id       TEXT    NOT NULL REFERENCES clientes(cliente_id),
        tipo             TEXT    NOT NULL,
        monto_propuesto  REAL    NOT NULL,
        plazo_meses      INTEGER NOT NULL,
        estado           TEXT    NOT NULL DEFAULT 'enviada',
        timestamp        TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS log_decisiones (
        log_id           TEXT PRIMARY KEY,
        cliente_id       TEXT NOT NULL REFERENCES clientes(cliente_id),
        nodo_origen      TEXT NOT NULL,
        tool_llamada     TEXT,
        argumentos_json  TEXT,
        resultado_json   TEXT,
        razonamiento     TEXT,
        status           TEXT NOT NULL DEFAULT 'ok',
        timestamp        TEXT NOT NULL
    )
    """,
]

# Índices para las consultas más frecuentes del agente
_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_contactos_cliente    ON contactos    (cliente_id, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_escalaciones_cliente ON escalaciones (cliente_id, estado)",
    "CREATE INDEX IF NOT EXISTS idx_propuestas_cliente   ON propuestas   (cliente_id, estado)",
    "CREATE INDEX IF NOT EXISTS idx_log_cliente          ON log_decisiones (cliente_id, timestamp DESC)",
]


# ---------------------------------------------------------------------------
# Helpers para seed
# ---------------------------------------------------------------------------

def _ts(delta_days: int = 0, delta_hours: int = 0) -> str:
    """Timestamp ISO relativo al momento actual. Garantiza datos 'recientes'."""
    dt = datetime.now() - timedelta(days=delta_days, hours=delta_hours)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _uid(prefix: str = "") -> str:
    """ID único corto de 8 caracteres."""
    return prefix + str(uuid.uuid4())[:8].upper()


# ---------------------------------------------------------------------------
# Seed — clientes
# ---------------------------------------------------------------------------
# 23 clientes en 5 etapas con EWS scores, montos y mora acordes a cada etapa.
# Diseño: variedad suficiente para que el agente tome decisiones distintas
# dependiendo del cliente que procese en cada demo.

_CLIENTES: list[tuple] = [
    # (cliente_id, nombre, ews_score, dias_mora, monto_adeudado, limite_credito, etapa)
    # --- PREVENTIVA (EWS detectó señal incipiente, mora 1-10 días) ---
    ("CLI001", "Ana García Reyes",       0.68, 3,   12_500.00, 80_000.00,  "preventiva"),
    ("CLI002", "Carlos Mendoza Ortiz",   0.71, 5,    8_750.00, 50_000.00,  "preventiva"),
    ("CLI003", "Sofía Herrera López",    0.67, 7,   19_800.00, 120_000.00, "preventiva"),
    ("CLI004", "Miguel Ángel Ruiz",      0.73, 10,   6_300.00, 40_000.00,  "preventiva"),
    ("CLI005", "Patricia Flores Vega",   0.70, 4,   15_200.00, 90_000.00,  "preventiva"),
    # --- TEMPRANA (mora 15-45 días, intentos sin respuesta) ---
    ("CLI006", "Roberto Jiménez Cruz",   0.78, 18,  22_400.00, 100_000.00, "temprana"),
    ("CLI007", "Laura Martínez Soto",    0.80, 25,  31_500.00, 150_000.00, "temprana"),
    ("CLI008", "Fernando Torres Gil",    0.76, 32,   9_800.00, 60_000.00,  "temprana"),
    ("CLI009", "Claudia Ramírez Díaz",   0.82, 40,  44_000.00, 200_000.00, "temprana"),
    ("CLI010", "Javier Morales Peña",    0.79, 22,  17_600.00, 80_000.00,  "temprana"),
    ("CLI011", "Diana Castillo Nava",    0.77, 38,  28_900.00, 130_000.00, "temprana"),
    # --- ADMINISTRATIVA (mora 60-90 días, múltiples contactos fallidos) ---
    ("CLI012", "Alejandro Vargas Ríos",  0.85, 65,  53_200.00, 200_000.00, "administrativa"),
    ("CLI013", "Mónica Delgado Mora",    0.83, 78,  36_700.00, 160_000.00, "administrativa"),
    ("CLI014", "Héctor Guzmán Lara",     0.87, 90,  61_400.00, 250_000.00, "administrativa"),
    ("CLI015", "Verónica Reyes Luna",    0.84, 72,  24_800.00, 120_000.00, "administrativa"),
    # --- JUDICIAL (mora 120-180 días, montos altos, requieren pausa autónoma) ---
    ("CLI016", "Arturo Medina Fuentes",  0.91, 135, 78_600.00, 300_000.00, "judicial"),
    ("CLI017", "Gabriela Ibáñez Ramos",  0.89, 150, 45_300.00, 180_000.00, "judicial"),
    ("CLI018", "Luis Cervantes Vidal",   0.93, 175, 92_100.00, 350_000.00, "judicial"),
    ("CLI019", "Rosa Pacheco Ávila",     0.88, 128, 33_700.00, 150_000.00, "judicial"),
    # --- CASTIGO (mora 180+ días, recuperación remota) ---
    ("CLI020", "Salvador Núñez Bravo",   0.95, 210, 18_500.00, 80_000.00,  "castigo"),
    ("CLI021", "Teresa Aguilar Rojas",   0.96, 245, 67_400.00, 250_000.00, "castigo"),
    ("CLI022", "Ernesto Velázquez Lima", 0.94, 320, 41_200.00, 180_000.00, "castigo"),
    ("CLI023", "Carmen Espinoza Solís",  0.97, 410, 83_600.00, 300_000.00, "castigo"),
]


def _seed_clientes(cur: sqlite3.Cursor) -> None:
    for cid, nombre, ews, mora, monto, limite, etapa in _CLIENTES:
        cur.execute("""
            INSERT INTO clientes
                (cliente_id, nombre, ews_score, dias_mora, monto_adeudado,
                 limite_credito, etapa_cobranza, fecha_alta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (cid, nombre, ews, mora, monto, limite, etapa, _ts(mora)))


# ---------------------------------------------------------------------------
# Seed — contactos
# ---------------------------------------------------------------------------
# Variedad deliberada:
#   - Preventiva: pocos contactos, al menos uno exitoso
#   - Temprana: múltiples intentos fallidos, alguna promesa no cumplida
#   - Administrativa: patrón claro de no-contacto (>=5 fallidos)
#   - Judicial: rechazos explícitos, bloqueos

_CONTACTOS: list[tuple] = [
    # (cliente_id, canal, resultado, delta_days, delta_hours, notas)

    # CLI001 preventiva — contacto reciente exitoso
    ("CLI001", "sms",      "no_contesto",  3, 2,  ""),
    ("CLI001", "llamada",  "contesto",     2, 4,  "Al tanto de la mora, promete regularizar esta semana"),

    # CLI003 preventiva — sin respuesta aún
    ("CLI003", "sms",      "no_contesto",  7, 3,  ""),

    # CLI006 temprana — patrón de no-contacto + promesa incumplida
    ("CLI006", "sms",      "no_contesto",  18, 6, ""),
    ("CLI006", "llamada",  "no_contesto",  14, 3, ""),
    ("CLI006", "whatsapp", "no_contesto",  10, 1, ""),
    ("CLI006", "llamada",  "contesto",      7, 5, "Dice que pagará la próxima semana, pide esperar"),
    ("CLI006", "sms",      "no_contesto",   3, 2, "No cumplió promesa de pago"),

    # CLI007 temprana — promesa de pago documentada, no cumplida
    ("CLI007", "email",    "no_contesto",  25, 8, ""),
    ("CLI007", "llamada",  "promesa_pago", 20, 4, "Promete pago parcial de $10,000 el viernes"),
    ("CLI007", "sms",      "no_contesto",  13, 2, "No realizó el pago prometido"),

    # CLI009 temprana — monto alto + rechazo explícito + bloqueo
    ("CLI009", "llamada",  "rechazo",      40, 6, "Cliente molesto, colgó antes de terminar"),
    ("CLI009", "email",    "no_contesto",  35, 3, ""),
    ("CLI009", "whatsapp", "rechazo",      28, 1, "Bloqueó el número de WhatsApp"),
    ("CLI009", "llamada",  "no_contesto",  21, 2, ""),
    ("CLI009", "sms",      "no_contesto",  14, 5, ""),

    # CLI012 administrativa — 6 intentos, un contacto útil en medio
    ("CLI012", "llamada",  "no_contesto",  65, 8, ""),
    ("CLI012", "sms",      "no_contesto",  58, 4, ""),
    ("CLI012", "email",    "contesto",     50, 2, "Solicita plan de pagos, pide tiempo para revisar"),
    ("CLI012", "llamada",  "no_contesto",  40, 6, ""),
    ("CLI012", "whatsapp", "no_contesto",  30, 3, ""),
    ("CLI012", "llamada",  "no_contesto",  20, 1, ""),

    # CLI013 administrativa — quita aceptada, historial de contacto previo
    ("CLI013", "llamada",  "contesto",     78, 5, "Acepta propuesta de quita del 30%"),
    ("CLI013", "email",    "no_contesto",  65, 3, ""),

    # CLI014 administrativa — promesas incumplidas repetidas
    ("CLI014", "llamada",  "promesa_pago", 90, 7, "Primera promesa de pago"),
    ("CLI014", "llamada",  "promesa_pago", 70, 4, "Segunda promesa, tampoco cumplida"),
    ("CLI014", "llamada",  "promesa_pago", 50, 2, "Tercera promesa incumplida"),
    ("CLI014", "whatsapp", "no_contesto",  35, 6, ""),
    ("CLI014", "llamada",  "no_contesto",  20, 3, ""),

    # CLI016 judicial — rechazo a gestión, referido a abogado
    ("CLI016", "llamada",  "rechazo",     135, 8, "Dice que tiene abogado, no quiere hablar"),
    ("CLI016", "email",    "no_contesto", 100, 4, ""),
    ("CLI016", "llamada",  "rechazo",      75, 2, "Colgó al identificar la institución"),

    # CLI018 judicial — monto altísimo, múltiples fallidos
    ("CLI018", "llamada",  "contesto",    175, 6, "Promesa inicial de pago parcial, nunca cumplida"),
    ("CLI018", "email",    "no_contesto", 140, 3, ""),
    ("CLI018", "llamada",  "no_contesto", 110, 1, ""),
    ("CLI018", "whatsapp", "rechazo",      90, 5, "Bloqueó número de WhatsApp corporativo"),
    ("CLI018", "llamada",  "no_contesto",  60, 2, ""),
    ("CLI018", "sms",      "no_contesto",  30, 4, ""),
]


def _seed_contactos(cur: sqlite3.Cursor) -> None:
    ultimo_por_cliente: dict[str, str] = {}

    for cid, canal, resultado, dd, dh, notas in _CONTACTOS:
        ts = _ts(dd, dh)
        cur.execute("""
            INSERT INTO contactos
                (contacto_id, cliente_id, canal, resultado, notas, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (_uid("CON"), cid, canal, resultado, notas, ts))

        # Rastrear timestamp más reciente por cliente
        if cid not in ultimo_por_cliente or ts > ultimo_por_cliente[cid]:
            ultimo_por_cliente[cid] = ts

    # Actualizar ultimo_contacto en clientes
    for cid, ts in ultimo_por_cliente.items():
        cur.execute(
            "UPDATE clientes SET ultimo_contacto = ? WHERE cliente_id = ?",
            (ts, cid)
        )


# ---------------------------------------------------------------------------
# Seed — escalaciones
# ---------------------------------------------------------------------------

_ESCALACIONES: list[tuple] = [
    # (cliente_id, nivel, motivo, estado, resuelto_por, fecha_res_delta, ts_delta)
    # CLI014 — escalada a supervisor por promesas repetidas, pendiente
    ("CLI014", "supervisor",
     "Tres promesas de pago incumplidas en 90 días de mora. Sin acuerdo viable.",
     "pendiente", None, None, 15),
    # CLI016 — escalada a legal por rechazo y monto alto, pendiente
    ("CLI016", "legal",
     "Monto > $50k, mora > 120 días, cliente con representación legal propia.",
     "pendiente", None, None, 30),
    # CLI018 — escalada a legal, monto crítico, pendiente
    ("CLI018", "legal",
     "Monto > $90k, cliente en posible proceso de insolvencia. Requiere análisis jurídico.",
     "pendiente", None, None, 45),
    # CLI021 — castigo contable ya resuelto (historial cerrado)
    ("CLI021", "castigo",
     "Mora de 245 días. Recuperación remota. Procede castigo contable.",
     "resuelto", "Gerente de Cartera", 10, 60),
]


def _seed_escalaciones(cur: sqlite3.Cursor) -> None:
    for cid, nivel, motivo, estado, resuelto_por, res_delta, ts_delta in _ESCALACIONES:
        fecha_res = _ts(res_delta) if res_delta is not None else None
        cur.execute("""
            INSERT INTO escalaciones
                (escalacion_id, cliente_id, nivel, motivo, estado,
                 resuelto_por, fecha_resolucion, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (_uid("ESC"), cid, nivel, motivo, estado, resuelto_por, fecha_res, _ts(ts_delta)))

        # Los casos pendientes bloquean el trigger automático
        if estado == "pendiente":
            cur.execute(
                "UPDATE clientes SET gestionado_humano = 1 WHERE cliente_id = ?",
                (cid,)
            )


# ---------------------------------------------------------------------------
# Seed — propuestas
# ---------------------------------------------------------------------------

_PROPUESTAS: list[tuple] = [
    # (cliente_id, tipo, monto, plazo_meses, estado, ts_delta)
    ("CLI012", "plan_pagos",       15_000.00, 6,  "enviada",   10),  # en espera de respuesta
    ("CLI013", "quita",            12_000.00, 1,  "aceptada",  20),  # cliente aceptó
    ("CLI014", "refinanciamiento", 30_000.00, 12, "rechazada", 25),  # cliente rechazó
    ("CLI015", "plan_pagos",        8_000.00, 4,  "enviada",   8),   # en espera
    ("CLI019", "quita",            10_000.00, 1,  "enviada",   12),  # bajo revisión legal
]


def _seed_propuestas(cur: sqlite3.Cursor) -> None:
    for cid, tipo, monto, plazo, estado, ts_delta in _PROPUESTAS:
        cur.execute("""
            INSERT INTO propuestas
                (propuesta_id, cliente_id, tipo, monto_propuesto,
                 plazo_meses, estado, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (_uid("PRO"), cid, tipo, monto, plazo, estado, _ts(ts_delta)))


# ---------------------------------------------------------------------------
# Seed — log_decisiones (muestra trazabilidad desde el arranque del demo)
# ---------------------------------------------------------------------------

def _seed_log(cur: sqlite3.Cursor) -> None:
    import json
    _LOG: list[tuple] = [
        ("CLI006", "decidir_accion", "registrar_contacto",
         json.dumps({"canal": "sms", "resultado": "no_contesto", "notas": ""}),
         json.dumps({"ok": True, "contacto_id": "CONDEMOA1"}),
         "Cliente sin respuesta en 5 intentos. Canal SMS como último intento antes de evaluar escalación.",
         "ok", 3),
        ("CLI012", "decidir_accion", "generar_propuesta",
         json.dumps({"tipo": "plan_pagos", "monto_propuesto": 15000, "plazo_meses": 6}),
         json.dumps({"ok": True, "propuesta_id": "PRODEMO01"}),
         "Cliente mostró disposición de pago en contacto previo. EWS score 0.85 indica riesgo moderado-alto. Plan de pagos es la opción más viable.",
         "ok", 10),
    ]
    for cid, nodo, tool, args, resultado, razon, status, ts_delta in _LOG:
        cur.execute("""
            INSERT INTO log_decisiones
                (log_id, cliente_id, nodo_origen, tool_llamada, argumentos_json,
                 resultado_json, razonamiento, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (_uid("LOG"), cid, nodo, tool, args, resultado, razon, status, _ts(ts_delta)))


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def inicializar_crm(force_reset: bool = False) -> None:
    """
    Crea las tablas del CRM e índices, y siembra datos sintéticos si está vacío.

    Idempotente: seguro llamarlo en cada arranque de la app. Si la DB ya tiene
    datos, no hace nada (a menos que force_reset=True).

    Args:
        force_reset: Elimina todos los datos y vuelve a sembrar. Útil para
                     resetear el demo desde Tab 6 o desde CLI.
    """
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    cur = con.cursor()

    # Crear tablas e índices (idempotente)
    for ddl in _DDL:
        cur.execute(ddl)
    for idx in _INDICES:
        cur.execute(idx)
    con.commit()

    if force_reset:
        # Orden inverso por foreign keys
        for tabla in ["log_decisiones", "propuestas", "escalaciones",
                      "contactos", "clientes"]:
            cur.execute(f"DELETE FROM {tabla}")
        con.commit()

    # Sembrar solo si la tabla clientes está vacía
    cur.execute("SELECT COUNT(*) FROM clientes")
    if cur.fetchone()[0] == 0:
        _seed_clientes(cur)
        _seed_contactos(cur)
        _seed_escalaciones(cur)
        _seed_propuestas(cur)
        _seed_log(cur)
        con.commit()

    con.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    reset = "--reset" in sys.argv
    inicializar_crm(force_reset=reset)

    con = sqlite3.connect(DB_PATH)
    print(f"\nCRM inicializado: {DB_PATH}")
    print("-" * 40)
    totales = {}
    for tabla in ["clientes", "contactos", "escalaciones", "propuestas", "log_decisiones"]:
        n = con.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
        totales[tabla] = n
        print(f"  {tabla:<20} {n:>3} registros")

    print("\nClientes por etapa:")
    for row in con.execute(
        "SELECT etapa_cobranza, COUNT(*) as n FROM clientes GROUP BY etapa_cobranza ORDER BY n DESC"
    ).fetchall():
        print(f"  {row[0]:<20} {row[1]:>3}")

    print("\nEscalaciones pendientes:")
    for row in con.execute(
        "SELECT cliente_id, nivel, motivo FROM escalaciones WHERE estado='pendiente'"
    ).fetchall():
        print(f"  {row[0]} | {row[1]} | {row[2][:60]}")
    con.close()
