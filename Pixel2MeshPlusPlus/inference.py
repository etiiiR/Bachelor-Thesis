# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import tflearn
import numpy as np
import pprint
import pickle
import shutil
import os
import glob
from pathlib import Path

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import execute
# from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict, load_demo_image
# from utils.visualize import plot_scatter


def process_all_pollen_folders(rgb_folders_path="./224x224rgb"):
    """
    Verarbeitet alle Pollenordner im 224x224rgb Verzeichnis.
    
    Args:
        rgb_folders_path: Pfad zum 224x224rgb Verzeichnis
    """
    # Verwende die Standard-Config wie im Original-Demo
    from modules.config import execute
    
    # Lade separate Configs für bessere Kontrolle
    try:
        # Versuche spezifische Configs zu laden
        import sys
        original_argv = sys.argv.copy()
        
        sys.argv = ['inference.py', '-f', 'cfgs/mvp2m_2_views.yaml']
        cfg_mvp2m = execute()
        
        sys.argv = ['inference.py', '-f', 'cfgs/p2mpp_2_views.yaml'] 
        cfg_p2mpp = execute()
        
        sys.argv = original_argv  # Restore original argv
        
        print(f"=> MVP2M Config geladen: {cfg_mvp2m.name}")
        print(f"=> P2MPP Config geladen: {cfg_p2mpp.name}")
        
    except Exception as e:
        print(f"=> Fehler beim Laden spezifischer Configs: {e}")
        print("=> Verwende Default-Config für beide Modelle")
        cfg_mvp2m = cfg_p2mpp = execute()
        print(f"=> Default Config geladen mit {cfg_mvp2m.num_input_images} views")
    # Finde alle Ordner im rgb Verzeichnis
    pollen_folders = [d for d in Path(rgb_folders_path).iterdir() if d.is_dir()]
    
    if not pollen_folders:
        print(f"Keine Ordner in {rgb_folders_path} gefunden!")
        return
    
    print(f"Gefunden {len(pollen_folders)} Pollenordner:")
    for folder in pollen_folders:
        print(f"  - {folder.name}")
    
    # Erstelle Ausgabeverzeichnis
    output_dir = Path("mesh_results")
    output_dir.mkdir(exist_ok=True)
    
    # Einmalige Modell-Initialisierung
    print('=> Initialisiere Modelle...')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    # TensorFlow Graph zurücksetzen
    tf.compat.v1.reset_default_graph()
    
    # Erstelle Placeholders
    num_blocks = 3
    num_supports = 2
    placeholders = {
        'features': tf.placeholder(tf.float32, shape=(None, 3), name='features'),
        'img_inp': tf.placeholder(tf.float32, shape=(cfg_mvp2m.num_input_images, 224, 224, 3), name='img_inp'),
        'labels': tf.placeholder(tf.float32, shape=(None, 6), name='labels'),
        'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
        'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
        'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)],
        'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks - 1)],
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),
        'sample_coord': tf.placeholder(tf.float32, shape=(43, 3), name='sample_coord'),
        'cameras': tf.placeholder(tf.float32, shape=(cfg_mvp2m.num_input_images, 5), name='Cameras'),
        'faces_triangle': [tf.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)],
        'sample_adj': [tf.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)],
    }
    # Füge num_input_images zu placeholders hinzu (wird von den Modellen benötigt)
    placeholders['num_input_images'] = cfg_mvp2m.num_input_images
    
    # Erstelle Modelle mit separaten Variable Scopes
    print('=> Erstelle Modelle...')
    
    # Verwende Config-Namen für korrekte Checkpoint-Pfade
    model1_dir = os.path.join(cfg_mvp2m.save_path, cfg_mvp2m.name, 'models')  # results/coarse_mvp2m_augmentation_2_inputs/models
    
    # Für P2MPP müssen wir die entsprechende Config laden oder hardcoded verwenden
    model2_dir = os.path.join(cfg_p2mpp.save_path, cfg_p2mpp.name, 'models')
    
    print(f"=> Model1 (MVP2M) path: {model1_dir}")
    print(f"=> Model2 (P2MPP) path: {model2_dir}")
    
    # Define models WITHOUT variable scopes to avoid variable collection issues
    # Instead, use tf.AUTO_REUSE to handle potential variable conflicts
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        # Force MVP2M to use 'meshnet' name to match checkpoint
        model1 = MVP2MNet(placeholders, logging=False, args=cfg_mvp2m, name='meshnet')
        model2 = P2MPPNet(placeholders, logging=False, args=cfg_p2mpp, name='meshnet')
    
    # TensorFlow Session
    print('=> Starte TensorFlow Session...')
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    
    # Lade Modell-Gewichte
    print('=> Lade Modell-Gewichte...')
    # Verwende test_epoch aus den Configs für korrekte Checkpoint-Steps
    mvp2m_step = cfg_mvp2m.test_epoch  # Normalerweise 200
    p2mpp_step = getattr(cfg_p2mpp, 'test_epoch', 30)  # Falls test_epoch nicht existiert, verwende 30
    
    print(f"=> Lade MVP2M Checkpoint: step {mvp2m_step}")
    print(f"=> Lade P2MPP Checkpoint: step {p2mpp_step}")
    
    model1.load(sess=sess, ckpt_path=model1_dir, step=mvp2m_step)
    model2.load(sess=sess, ckpt_path=model2_dir, step=p2mpp_step)
    
    # P2MPP benötigt die CNN-Gewichte von MVP2M - das ist kritisch!
    print("=> Lade CNN-Gewichte fuer P2MPP von MVP2M...")
    try:
        cnn_path = getattr(cfg_p2mpp, 'pre_trained_cnn_path', model1_dir)
        cnn_step = getattr(cfg_p2mpp, 'cnn_step', mvp2m_step)
        print(f"=> CNN-Pfad: {cnn_path}, Step: {cnn_step}")
        model2.loadcnn(sess=sess, ckpt_path=cnn_path, step=cnn_step)
    except Exception as e:
        print(f"=> CNN-Loading fehlgeschlagen: {e}")
        print("=> Verwende bereits geladene MVP2M CNN-Gewichte")
    
    # Lade Ellipsoid-Daten
    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    feed_dict = construct_feed_dict(pkl, placeholders)
    
    # Verarbeite alle Ordner
    tflearn.is_training(False, sess)
    successful_count = 0
    
    for i, pollen_folder in enumerate(pollen_folders, 1):
        print(f"\n=> [{i}/{len(pollen_folders)}] Verarbeite {pollen_folder.name}")
        
        # Prüfe ob die benötigten Dateien existieren
        img1_path = pollen_folder / "000000.png"
        img2_path = pollen_folder / "000001.png"
        cameras_path = pollen_folder / "cameras.txt"
        
        if not all([img1_path.exists(), img2_path.exists(), cameras_path.exists()]):
            print(f"  ❌ Überspringe: Fehlende Dateien")
            continue
        
        try:
            # Lade Bilder und Kamera-Daten
            img_paths = [str(img1_path), str(img2_path)]
            img_all_view = load_demo_image(img_paths)
            cameras = np.loadtxt(str(cameras_path))
            
            print(f"  [IMG] Bilder geladen: {img_all_view.shape}")
            
            # Führe Inferenz durch
            # Stage 1 - Verwende separate feed_dict um die Original-Struktur zu bewahren
            stage1_feed_dict = feed_dict.copy()
            stage1_feed_dict.update({placeholders['img_inp']: img_all_view})
            stage1_feed_dict.update({placeholders['labels']: np.zeros([10, 6])})
            stage1_feed_dict.update({placeholders['cameras']: cameras})
            stage1_out3 = sess.run(model1.output3, stage1_feed_dict)
            
            # Stage 2 - Verwende separate feed_dict mit stage1 output als features
            stage2_feed_dict = feed_dict.copy()
            stage2_feed_dict.update({placeholders['img_inp']: img_all_view})
            stage2_feed_dict.update({placeholders['labels']: np.zeros([10, 6])})
            stage2_feed_dict.update({placeholders['cameras']: cameras})
            stage2_feed_dict.update({placeholders['features']: stage1_out3})
            vert = sess.run(model2.output2l, stage2_feed_dict)
            vert = np.hstack((np.full([vert.shape[0], 1], 'v'), vert))
            face = np.loadtxt('data/face3.obj', dtype='|S32')
            mesh = np.vstack((vert, face))
            
            # Speichere Ergebnis
            output_file = output_dir / f"{pollen_folder.name}.obj"
            np.savetxt(output_file, mesh, fmt='%s', delimiter=' ')
            
            successful_count += 1
            print(f"  [OK] Mesh gespeichert: {output_file}")
            
        except Exception as e:
            print(f"  [ERROR] Fehler: {e}")
            continue
    
    # Session schließen
    sess.close()
    print(f"\n🎉 Verarbeitung abgeschlossen! {successful_count}/{len(pollen_folders)} Meshes erstellt.")


def main():
    """
    Hauptfunktion - verarbeitet alle Pollenordner im 224x224rgb Verzeichnis.
    """
    print('=> Starte Batch-Verarbeitung aller Pollen')
    
    # Verarbeite alle Pollenordner
    process_all_pollen_folders()
    
    print('\n=> Batch-Verarbeitung abgeschlossen!')


if __name__ == '__main__':
    print('=> Starte Batch-Verarbeitung aller Pollen')
    main()
