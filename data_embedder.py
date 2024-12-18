# data_embedder.py

import os
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

class DataEmbedder:
    """데이터를 준비하고 임베딩을 생성/저장하는 클래스입니다."""
    
    def __init__(self, base_dir: str = "data_store"):
        """
        데이터 임베더를 초기화합니다.
        
        Args:
            base_dir (str): 데이터와 임베딩을 저장할 기본 디렉토리
        """
        load_dotenv()
        
        # 기본 디렉토리 구조 설정
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "cache"
        self.db_dir = self.base_dir / "chromadb"
        
        # 필요한 디렉토리 생성
        self.base_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_dir.mkdir(exist_ok=True)
        
        # ChromaDB 클라이언트 초기화 (영구 저장소 사용)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        
        # OpenAI 임베딩 함수 설정
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

    def _get_or_create_collection(self, collection_name: str = "mission_dataset"):
        """컬렉션을 가져오거나 새로 생성합니다."""
        try:
            # 기존 컬렉션 불러오기 시도
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"기존 컬렉션을 불러왔습니다: {collection_name}")
        except (ValueError, chromadb.errors.InvalidCollectionException):
        # 컬렉션이 없으면 새로 생성
            collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        print(f"새로운 컬렉션을 생성했습니다: {collection_name}")
        
        return collection

    def process_and_embed_data(self, file_paths: dict):
        """
        여러 CSV 파일의 데이터를 처리하고 임베딩합니다.
        
        Args:
            file_paths (dict): 언어별 CSV 파일 경로 딕셔너리
                             예: {"Java": "path/to/java.csv", ...}
        """
        # 컬렉션 준비
        collection = self._get_or_create_collection()
        
        for language, file_path in file_paths.items():
            try:
                # 캐시 파일 경로
                cache_path = self.cache_dir / f"{Path(file_path).stem}_cache.csv"
                
                # 데이터 로드 (캐시 활용)
                if cache_path.exists():
                    print(f"캐시된 데이터를 불러옵니다: {language}")
                    df = pd.read_csv(cache_path)
                else:
                    print(f"새로운 데이터를 로드합니다: {language}")
                    df = pd.read_csv(file_path)
                    # 캐시 저장
                    df.to_csv(cache_path, index=False)
                
                # 기존 데이터 확인
                existing_ids = set(collection.get()["ids"])
                
                # 데이터 임베딩 및 저장
                for idx, row in df.iterrows():
                    doc_id = f"{language}_{idx}"
                    
                    # 이미 저장된 데이터는 건너뛰기
                    if doc_id in existing_ids:
                        continue
                    
                    # 문서 텍스트 생성
                    document = f"Topic: {row['topic']}\nDescription: {row['topic_description']}"
                    
                    # ChromaDB에 데이터 추가
                    collection.add(
                        documents=[document],
                        metadatas=[{
                            #"language": language,
                            "language": language.upper(),
                            "topic": row["topic"],
                            "description": row["topic_description"],
                            "difficulty_level": row["difficulty_level"]
                        }],
                        #ids=[doc_id]
                        ids=[f"{language.upper()}_{idx}"]  # ID도 일관되게
                    )
                
                print(f"{language} 데이터 처리 완료: {len(df)} 행")
                
            except Exception as e:
                print(f"{language} 데이터 처리 중 오류 발생: {e}")

def prepare_data():
    """데이터 준비 및 임베딩 실행 함수"""
    # 기본 설정
    base_dir = "data_store"
    data_dir = "Devita_data"
    
    # 지원하는 프로그래밍 언어와 파일 경로
    file_paths = {
        lang: f"{data_dir}/Mission_Dataset_{lang}.csv"
        for lang in ["JAVA", "PYTHON", "JAVASCRIPT", "REACT", "SPRING", "DOCKER"]
    }
    
    # 데이터 임베더 초기화 및 실행
    embedder = DataEmbedder(base_dir)
    embedder.process_and_embed_data(file_paths)

if __name__ == "__main__":
    prepare_data()