import os
import copy
from dataclasses import dataclass, field
from typing import List

# --- DEFINICIÓN DE ESTRUCTURAS DE DATOS ---

@dataclass
class Asignatura:
    """Representa una asignatura con sus propiedades."""
    id: int
    nombre: str
    horas: int
    curso: str
    tipo: str
    preferencia_javier: int

    def __hash__(self):
        return self.id

    def __str__(self):
        return f"{self.nombre} ({self.horas}h)"

# --- DATOS INICIALES DEL PROBLEMA ---

ASIGNATURAS_BASE = [
    Asignatura(1, "Sistemas informáticos", 6, "1º DAW", "Común", 7),
    Asignatura(2, "Diseño de interfaces web", 4, "2º DAW", "Común", 5),
    Asignatura(3, "Desarrollo Web en entorno cliente", 6, "2º DAW", "Común", 1),
    Asignatura(4, "Optativa 2º DAW", 3, "2º DAW", "Optativa", 2),
    Asignatura(5, "Digitalización", 1, "2º DAW", "Común", 3),
    Asignatura(6, "FFE", 3, "2º DAW", "FFE", 9),
    Asignatura(7, "Implantación de sistemas operativos", 7, "1º ASIR", "Común", 8),
    Asignatura(8, "Optativa 2º ASIR", 3, "2º ASIR", "Optativa", 4),
    Asignatura(9, "Administración de sistemas operativos", 5, "2º ASIR", "Común", 6),
]

# --- UTILIDADES Y LÓGICA DEL JUEGO --

class Colors:
    """Clase para almacenar los códigos de color ANSI."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

# --- LÓGICA DEL ESTADO DEL JUEGO ---

@dataclass
class EstadoJuego:
    """Almacena el estado completo de una partida en un momento dado."""
    asignaturas_restantes: List[Asignatura] = field(default_factory=lambda: copy.deepcopy(ASIGNATURAS_BASE))
    javier_asignaturas: List[Asignatura] = field(default_factory=list)
    fernando_asignaturas: List[Asignatura] = field(default_factory=list)
    javier_horas: int = 0
    fernando_horas: int = 0
    turno: str = "Javier"
    
    def copy(self):
        """Crea una copia profunda del estado actual."""
        return copy.deepcopy(self)

    def get_posibles_movimientos(self):
        """Devuelve las asignaturas que se pueden elegir en el turno actual."""
        movimientos_validos = []
        
        jugador_actual_asignaturas = self.javier_asignaturas if self.turno == "Javier" else self.fernando_asignaturas
        jugador_ya_tiene_2daw = any(a.curso == "2º DAW" and a.tipo != "FFE" for a in jugador_actual_asignaturas)
        
        daw_restantes_sin_ffe = [a for a in self.asignaturas_restantes if a.curso == "2º DAW" and a.tipo != "FFE"]

        for asig in self.asignaturas_restantes:
            if asig.tipo == "FFE":
                # Solo se puede coger FFE si ya se tiene una de 2º DAW o si quedan disponibles para coger en el futuro.
                if not jugador_ya_tiene_2daw and not daw_restantes_sin_ffe:
                    continue # No se puede elegir FFE, es una jugada ilegal garantizada

            movimientos_validos.append(asig)

        return movimientos_validos

    def elegir_asignatura(self, asignatura: Asignatura):
        """Aplica un movimiento al estado actual y devuelve un nuevo estado."""
        nuevo_estado = self.copy()
        asig_a_mover = next(a for a in nuevo_estado.asignaturas_restantes if a.id == asignatura.id)
        
        if nuevo_estado.turno == "Javier":
            nuevo_estado.javier_asignaturas.append(asig_a_mover)
            nuevo_estado.javier_horas += asig_a_mover.horas
            nuevo_estado.turno = "Fernando"
        else:
            nuevo_estado.fernando_asignaturas.append(asig_a_mover)
            nuevo_estado.fernando_horas += asig_a_mover.horas
            nuevo_estado.turno = "Javier"
            
        nuevo_estado.asignaturas_restantes.remove(asig_a_mover)

        # Si un jugador alcanza un cupo válido, el otro se queda con el resto
        if nuevo_estado.javier_horas in [19]:
            horas_restantes = sum(a.horas for a in nuevo_estado.asignaturas_restantes)
            if nuevo_estado.fernando_horas + horas_restantes == 38 - nuevo_estado.javier_horas:
                nuevo_estado.fernando_asignaturas.extend(nuevo_estado.asignaturas_restantes)
                nuevo_estado.fernando_horas += horas_restantes
                nuevo_estado.asignaturas_restantes = []

        if nuevo_estado.fernando_horas in [19]:
            horas_restantes = sum(a.horas for a in nuevo_estado.asignaturas_restantes)
            if nuevo_estado.javier_horas + horas_restantes == 38 - nuevo_estado.fernando_horas:
                nuevo_estado.javier_asignaturas.extend(nuevo_estado.asignaturas_restantes)
                nuevo_estado.javier_horas += horas_restantes
                nuevo_estado.asignaturas_restantes = []

        return nuevo_estado

    def is_terminal(self):
        """Comprueba si el juego ha terminado."""
        return not self.asignaturas_restantes

    def evaluar_resultado(self):
        """
        Evalúa un estado terminal desde la perspectiva de Javier.
        -  2: Victoria (Fernando coge FFE en un reparto válido).
        -  1: Victoria (Bloqueo de horas, reparto inválido).
        - -1: Empate (reparto válido 19-19, nadie coge FFE).
        - -2: Derrota (Javier coge FFE en un reparto válido).
        """
        total_horas = self.javier_horas + self.fernando_horas
        reparto_valido = (self.javier_horas, self.fernando_horas) in [(19, 19)]

        # 1. La condición más importante es si el reparto de horas es válido.
        # Si no es válido, es un bloqueo, sin importar quién tenga FFE.
        if total_horas != 38 or not reparto_valido:
            return 1  # Victoria por Bloqueo

        # --- A partir de aquí, sabemos que el reparto de horas ES VÁLIDO ---

        javier_tiene_ffe = any(a.tipo == "FFE" for a in self.javier_asignaturas)
        fernando_tiene_ffe = any(a.tipo == "FFE" for a in self.fernando_asignaturas)

        # 2. Comprobar la regla FFE. Como el reparto de horas es válido, una
        # violación de la regla FFE es una derrota para quien la comete.
        if javier_tiene_ffe:
            if not any(a.curso == "2º DAW" and a.tipo != "FFE" for a in self.javier_asignaturas):
                return -2 # Derrota: Javier viola la regla FFE
        
        if fernando_tiene_ffe:
            if not any(a.curso == "2º DAW" and a.tipo != "FFE" for a in self.fernando_asignaturas):
                return 2 # Victoria: Fernando viola la regla FFE

        # 3. Si el reparto es válido y no hay violaciones, determinar ganador por FFE.
        if fernando_tiene_ffe:
            return 2  # Victoria para Javier
        
        if javier_tiene_ffe:
            return -2 # Derrota para Javier

        # 4. Si el reparto es válido y nadie tiene FFE, es un empate.
        return -1 # Empate

# --- ALGORITMOS DE ANÁLISIS (MINIMAX Y FUERZA BRUTA) ---

memo_minimax = {}
def minimax(estado: EstadoJuego, es_turno_javier: bool):
    """Algoritmo Minimax para determinar el mejor movimiento."""
    estado_tupla = (tuple(sorted(a.id for a in estado.asignaturas_restantes)),
                    tuple(sorted(a.id for a in estado.javier_asignaturas)),
                    tuple(sorted(a.id for a in estado.fernando_asignaturas)),
                    es_turno_javier)
    if estado_tupla in memo_minimax:
        return memo_minimax[estado_tupla]

    if estado.is_terminal():
        return estado.evaluar_resultado()

    movimientos = estado.get_posibles_movimientos()
    if not movimientos:
        return estado.evaluar_resultado()

    if es_turno_javier:
        max_eval = -float('inf')
        for mov in movimientos:
            nuevo_estado = estado.elegir_asignatura(mov)
            eval = minimax(nuevo_estado, False)
            max_eval = max(max_eval, eval)
        memo_minimax[estado_tupla] = max_eval
        return max_eval
    else: # Turno de Fernando (minimizador)
        min_eval = float('inf')
        for mov in movimientos:
            nuevo_estado = estado.elegir_asignatura(mov)
            eval = minimax(nuevo_estado, True)
            min_eval = min(min_eval, eval)
        memo_minimax[estado_tupla] = min_eval
        return min_eval

memo_fuerza_bruta = {}
def simular_fuerza_bruta(estado: EstadoJuego):
    """
    Simula todas las partidas posibles y devuelve un recuento de resultados.
    Devuelve: (victorias_ffe, derrotas_ffe, bloqueos, total_partidas)
    """
    estado_tupla = (tuple(sorted(a.id for a in estado.asignaturas_restantes)),
                    tuple(sorted(a.id for a in estado.javier_asignaturas)),
                    tuple(sorted(a.id for a in estado.fernando_asignaturas)))
    if estado_tupla in memo_fuerza_bruta:
        return memo_fuerza_bruta[estado_tupla]

    if estado.is_terminal() or not estado.get_posibles_movimientos():
        resultado = estado.evaluar_resultado()
        if resultado == 2:  # Victoria FFE
            return (1, 0, 0, 1)
        elif resultado == 1:  # Bloqueo
            return (0, 0, 1, 1)
        elif resultado == -2: # Derrota FFE
            return (0, 1, 0, 1)
        else: # Empate o similar
            return (0, 0, 0, 1)

    total_victorias_ffe = 0
    total_derrotas_ffe = 0
    total_bloqueos = 0
    total_partidas = 0

    for mov in estado.get_posibles_movimientos():
        nuevo_estado = estado.elegir_asignatura(mov)
        victorias, derrotas, bloqueos, partidas = simular_fuerza_bruta(nuevo_estado)
        total_victorias_ffe += victorias
        total_derrotas_ffe += derrotas
        total_bloqueos += bloqueos
        total_partidas += partidas
    
    resultado_final = (total_victorias_ffe, total_derrotas_ffe, total_bloqueos, total_partidas)
    memo_fuerza_bruta[estado_tupla] = resultado_final
    return resultado_final

# --- INTERFAZ PRINCIPAL ---

def limpiar_pantalla():
    """Limpia la consola."""
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_estado(estado: EstadoJuego):
    """Muestra el estado actual del juego de forma clara."""
    print("="*50)
    print("ESTADO ACTUAL DE LA ELECCIÓN")
    print("-"*50)
    print(f"Turno de: {estado.turno}")
    print("\n--- Javier ---")
    print(f"Horas: {estado.javier_horas}")
    for asig in estado.javier_asignaturas:
        print(f"  - {asig}")
    
    print("\n--- Fernando ---")
    print(f"Horas: {estado.fernando_horas}")
    for asig in estado.fernando_asignaturas:
        print(f"  - {asig}")
    print("="*50 + "\n")


def main():
    """Bucle principal del programa interactivo."""
    historial_estados = [EstadoJuego()]

    while True:
        estado = historial_estados[-1]

        if estado.is_terminal():
            break

        limpiar_pantalla()
        mostrar_estado(estado)
        
        movimientos_posibles = estado.get_posibles_movimientos()

        if not movimientos_posibles:
            print("No hay más movimientos posibles. Fin del juego.")
            break

        asignatura_elegida = None

        if estado.turno == "Javier":
            print("--- ASIGNATURAS DISPONIBLES PARA JAVIER (Análisis de jugada) ---")
            opciones_analizadas = []
            for mov in movimientos_posibles:
                estado_futuro = estado.elegir_asignatura(mov)
                
                resultado_minimax = minimax(estado_futuro, False)
                v_ffe, d_ffe, bloq, totales = simular_fuerza_bruta(estado_futuro)
                
                opciones_analizadas.append({
                    "asignatura": mov,
                    "minmax": resultado_minimax,
                    "fb_victorias_ffe": v_ffe,
                    "fb_derrotas_ffe": d_ffe,
                    "fb_bloqueos": bloq,
                    "fb_totales": totales,
                    "preferencia": mov.preferencia_javier
                })
            
            opciones_analizadas.sort(key=lambda x: (-x['minmax'], x['preferencia'], -(x['fb_victorias_ffe'] + x['fb_bloqueos'])/x['fb_totales'] if x['fb_totales'] > 0 else 0))

            for i, opcion in enumerate(opciones_analizadas):
                asig = opcion['asignatura']
                mm_val = opcion['minmax']
                v_ffe = opcion['fb_victorias_ffe']
                d_ffe = opcion['fb_derrotas_ffe']
                bloq = opcion['fb_bloqueos']
                totales = opcion['fb_totales']

                r_v = v_ffe / totales if totales > 0 else 0
                r_d = d_ffe / totales if totales > 0 else 0
                r_b = bloq / totales if totales > 0 else 0
                
                color = ""
                if mm_val == 2: color = Colors.GREEN
                elif mm_val == 1: color = Colors.YELLOW
                elif mm_val <= 0: color = Colors.RED
                
                mm_str = f"minmax {mm_val: >2}"
                fb_str = f"G: {r_v:.2f} | P: {r_d:.2f} | B: {r_b:.2f}"
                print(f"{color}{i+1}. {asig.nombre:<40} ({asig.horas}h, {asig.curso}) | {mm_str} | FB: {fb_str}{Colors.RESET}")

            while True:
                choice = input(f"\nElige la asignatura para Javier (1-{len(opciones_analizadas)}) o 'd' para deshacer: ")
                if choice.lower() == 'd':
                    if len(historial_estados) > 1:
                        historial_estados.pop()
                        asignatura_elegida = None
                    else:
                        print("No hay movimientos que deshacer.")
                    break
                try:
                    eleccion_idx = int(choice) - 1
                    if 0 <= eleccion_idx < len(opciones_analizadas):
                        asignatura_elegida = opciones_analizadas[eleccion_idx]['asignatura']
                        break
                    else:
                        print("Número fuera de rango.")
                except ValueError:
                    print("Entrada no válida. Introduce un número o 'd'.")

        else: # Turno de Fernando
            print("--- ASIGNATURAS DISPONIBLES PARA FERNANDO ---")
            movimientos_posibles.sort(key=lambda x: x.nombre)
            for i, asig in enumerate(movimientos_posibles):
                print(f"{i+1}. {asig.nombre} ({asig.horas}h, {asig.curso})")
            
            while True:
                choice = input(f"\nElige la asignatura para Fernando (1-{len(movimientos_posibles)}) o 'd' para deshacer: ")
                if choice.lower() == 'd':
                    if len(historial_estados) > 1:
                        historial_estados.pop()
                        asignatura_elegida = None
                    else:
                        print("No hay movimientos que deshacer.")
                    break
                try:
                    eleccion_idx = int(choice) - 1
                    if 0 <= eleccion_idx < len(movimientos_posibles):
                        asignatura_elegida = movimientos_posibles[eleccion_idx]
                        break
                    else:
                        print("Número fuera de rango.")
                except ValueError:
                    print("Entrada no válida. Introduce un número o 'd'.")
        
        if asignatura_elegida:
            nuevo_estado = estado.elegir_asignatura(asignatura_elegida)
            historial_estados.append(nuevo_estado)
        elif choice.lower() != 'd':
            continue

    final_estado = historial_estados[-1]
    limpiar_pantalla()
    print("="*50)
    print("JUEGO TERMINADO - REPARTO FINAL")
    print("="*50)
    mostrar_estado(final_estado)
    
    resultado_final = final_estado.evaluar_resultado()
    print("\n--- CONCLUSIÓN ---")
    if resultado_final == 2:
        print("¡VICTORIA PARA JAVIER! Fernando ha tenido que elegir FFE o ha realizado una jugada ilegal con ella.")
    elif resultado_final == 1:
        print("¡VICTORIA PARA JAVIER! Se ha producido un bloqueo en el reparto de horas.")
    elif resultado_final == -2:
        print("DERROTA PARA JAVIER. Ha tenido que elegir FFE o ha realizado una jugada ilegal con ella.")
    else: # Empates (0 y -1)
        print("EMPATE. El reparto es válido y nadie ha elegido FFE.")


if __name__ == "__main__":
    main()
