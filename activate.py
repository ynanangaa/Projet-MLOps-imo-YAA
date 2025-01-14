import os
import subprocess
import sys


def main():
    # Vérifier si Poetry est installé
    try:
        subprocess.run(
            ["poetry", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print(
            "Poetry n'est pas installé. Veuillez installer Poetry avant de " +\
                "continuer."
        )
        sys.exit(1)

    # Vérifier si le fichier pyproject.toml existe
    if not os.path.exists("pyproject.toml"):
        print(
            "Le fichier pyproject.toml est introuvable. Êtes-vous dans le " +\
                "répertoire racine du projet ?"
        )
        sys.exit(1)

    # Activer l'environnement virtuel de Poetry
    print("Activation de l'environnement virtuel Poetry...")
    subprocess.run(["poetry", "install"], check=True)

    # Définir PYTHONPATH pour inclure le répertoire racine du projet
    os.environ["PYTHONPATH"] = os.getcwd()
    print(f"PYTHONPATH défini à : {os.environ['PYTHONPATH']}")

    # Activer le shell Poetry
    print("Pour activer le shell Poetry, exécutez :")
    print("poetry shell")


if __name__ == "__main__":
    main()
