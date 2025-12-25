"""
Script untuk upload model BERT dan DistilBERT ke Hugging Face Hub
Jalankan setelah login: huggingface-cli login
"""

from huggingface_hub import HfApi, create_repo
import os

# Ganti dengan username Hugging Face Anda
HF_USERNAME = "Rahma13"  # GANTI INI!

# Inisialisasi API
api = HfApi()

def upload_model(model_path, repo_name, model_type="bert"):
    """
    Upload model ke Hugging Face Hub
    
    Args:
        model_path: Path ke folder model lokal
        repo_name: Nama repository di HF (contoh: "spam-detection-bert")
        model_type: Tipe model untuk README
    """
    
    # Buat repository ID
    repo_id = f"{HF_USERNAME}/{repo_name}"
    
    print(f"\n{'='*60}")
    print(f"Uploading {model_type.upper()} model to: {repo_id}")
    print(f"{'='*60}\n")
    
    try:
        # Buat repository (skip jika sudah ada)
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False  # Set True jika ingin private
        )
        print(f"✅ Repository created/verified: {repo_id}")
        
        # Upload semua file dalam folder
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {model_type} spam detection model"
        )
        
        print(f"✅ Successfully uploaded model to: https://huggingface.co/{repo_id}")
        print(f"\n📝 Untuk load model di app.py, gunakan:")
        print(f'   AutoModelForSequenceClassification.from_pretrained("{repo_id}")')
        
    except Exception as e:
        print(f"❌ Error uploading {model_type}: {str(e)}")
        return False
    
    return True

def main():
    print("\n🚀 Hugging Face Model Upload Script")
    print("="*60)
    
    # Validasi username
    if HF_USERNAME == "your-username":
        print("❌ ERROR: Silakan ganti 'your-username' dengan username Hugging Face Anda!")
        print("   Edit file ini dan ubah variabel HF_USERNAME di baris 10")
        return
    
    # Path ke model
    bert_path = "src/models/bert"
    distilbert_path = "src/models/distilbert"
    
    # Cek apakah folder ada
    if not os.path.exists(bert_path):
        print(f"❌ Folder tidak ditemukan: {bert_path}")
        return
    
    if not os.path.exists(distilbert_path):
        print(f"❌ Folder tidak ditemukan: {distilbert_path}")
        return
    
    # Upload BERT
    print("\n[1/2] Uploading BERT model...")
    bert_success = upload_model(
        model_path=bert_path,
        repo_name="spam-detection-bert",
        model_type="BERT"
    )
    
    # Upload DistilBERT
    print("\n[2/2] Uploading DistilBERT model...")
    distilbert_success = upload_model(
        model_path=distilbert_path,
        repo_name="spam-detection-distilbert",
        model_type="DistilBERT"
    )
    
    # Summary
    print("\n" + "="*60)
    print("📊 Upload Summary:")
    print("="*60)
    print(f"BERT:       {'✅ Success' if bert_success else '❌ Failed'}")
    print(f"DistilBERT: {'✅ Success' if distilbert_success else '❌ Failed'}")
    
    if bert_success and distilbert_success:
        print("\n🎉 Semua model berhasil diupload!")
        print("\n📝 Next Steps:")
        print("1. Update app.py untuk load model dari Hugging Face")
        print("2. Hapus file .safetensors dari git:")
        print("   git rm --cached src/models/bert/model.safetensors")
        print("   git rm --cached src/models/distilbert/model.safetensors")
        print("3. Commit dan push ke GitHub")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
