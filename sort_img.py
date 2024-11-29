import os
import shutil

def organize_images_with_images_folder_handling(no_color_datasets, color_datasets, output_dir="combined_images"):
    """
    Erstellt zwei Ordner, `no_color_img_comb` und `color_img_comb`, 
    und kopiert Bilder aus den übergebenen Datensätzen in die entsprechenden Ordner.
    Der erste Datensatz hat einen speziellen Ordner `images`, der berücksichtigt werden soll,
    während alle anderen Unterordner rekursiv verarbeitet werden.

    Args:
        no_color_datasets (list): Liste der Pfade zu den Schwarz-Weiß-Datensätzen.
        color_datasets (list): Liste der Pfade zu den Farbdatensätzen.
        output_dir (str): Basisverzeichnis für die Ausgabedaten.
        
    Returns:
        tuple: Pfade zu den Ordnern `no_color_img_comb` und `color_img_comb`.
    """
    # Pfade für die kombinierten Ordner
    no_color_output = os.path.join(output_dir, "no_color_img_comb")
    color_output = os.path.join(output_dir, "color_img_comb")
    
    # Ordner erstellen, falls nicht vorhanden
    os.makedirs(no_color_output, exist_ok=True)
    os.makedirs(color_output, exist_ok=True)
    
    # Funktion zum Kopieren mit Kontrolle über Unterordner
    def copy_images(dataset_path, output_folder, label, recursive=True, subfolder=""):
        if recursive:
            # Rekursiv durch alle Unterordner
            for root, _, files in os.walk(dataset_path):
                # Wenn der aktuelle Ordner ein Unterordner von 'images' ist, sollte er kopiert werden
                if subfolder and os.path.basename(root) == subfolder:
                    print(f"Verarbeite Ordner: {root} ({label})")
                    for file_name in files:
                        source_path = os.path.join(root, file_name)
                        if os.path.isfile(source_path):
                            shutil.copy(source_path, output_folder)
                            print(f"  Kopiert: {file_name} → {output_folder}")
                # Ansonsten alle anderen Unterordner ignorieren
                elif not subfolder:
                    print(f"Verarbeite Ordner: {root} ({label})")
                    for file_name in files:
                        source_path = os.path.join(root, file_name)
                        if os.path.isfile(source_path):
                            shutil.copy(source_path, output_folder)
                            print(f"  Kopiert: {file_name} → {output_folder}")
        else:
            # Nur Dateien aus dem Hauptverzeichnis kopieren
            print(f"Verarbeite Hauptordner: {dataset_path} ({label})")
            for file_name in os.listdir(dataset_path):
                source_path = os.path.join(dataset_path, file_name)
                if os.path.isfile(source_path):
                    shutil.copy(source_path, output_folder)
                    print(f"  Kopiert: {file_name} → {output_folder}")
    
    # Bilder aus den Schwarz-Weiß-Datensätzen kopieren
    print("Beginne mit Schwarz-Weiß-Bildern...")
    for idx, dataset_path in enumerate(no_color_datasets):
        if idx == 0:
            # Für den ersten Datensatz (mit 'images'-Ordner) rekursiv nur den 'images'-Ordner berücksichtigen
            copy_images(dataset_path, no_color_output, "Schwarz-Weiß", recursive=True, subfolder="images")
        else:
            # Für andere Datensätze alle Unterordner berücksichtigen
            copy_images(dataset_path, no_color_output, "Schwarz-Weiß", recursive=True)
    
    # Bilder aus den Farbdatensätzen kopieren
    print("\nBeginne mit Farbbildern...")
    for idx, dataset_path in enumerate(color_datasets):
        if idx == 0:
            # Für den ersten Datensatz (mit 'images'-Ordner) rekursiv nur den 'images'-Ordner berücksichtigen
            copy_images(dataset_path, color_output, "Farbe", recursive=True, subfolder="images")
        else:
            # Für andere Datensätze alle Unterordner berücksichtigen
            copy_images(dataset_path, color_output, "Farbe", recursive=True)
    
    print("\nKopieren abgeschlossen.")
    return no_color_output, color_output


no_color_datasets = [
    r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\fashion_dataset_no_color\fashion-dataset",  # Enthält den Unterordner "images"
    r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\gender_dataset_no_color",  # Weitere Unterordner
    r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\natural_images_no_color"   # Weitere Unterordner
]

color_datasets = [
    r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\fashion_dataset_color",  # Enthält den Unterordner "images"
    r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\gender_dataset_color",  # Weitere Unterordner
    r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\natural_images_color"   # Weitere Unterordner
]

no_color_path, color_path = organize_images_with_images_folder_handling(no_color_datasets, color_datasets)

print(f"Schwarz-Weiß-Bilder gespeichert in: {no_color_path}")
print(f"Farbige Bilder gespeichert in: {color_path}")