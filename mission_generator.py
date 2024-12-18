import os
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import random
import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import InvalidCollectionException
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith import Client
from dotenv import load_dotenv
import json

# 카테고리와 토픽 정의
CATEGORIES = {
    "CS": ["DATA_STRUCTURE", "ALGORITHM", "COMPUTER_ARCHITECTURE", "NETWORK", "OPERATING_SYSTEM", "DATABASE"],
    "LANGUAGE": ["JAVA", "JAVASCRIPT", "PYTHON"],
    "TOOL": ["SPRING", "REACT", "PYTORCH", "DOCKER"]
}

# ChromaDB에 있는 데이터셋
CHROMADB_TOPICS = ["JAVA", "PYTHON", "JAVASCRIPT", "REACT", "SPRING", "DOCKER"]

class LangSmithLogger:
    """LangSmith를 사용하여 미션 생성 과정을 로깅하는 클래스"""
    
    def __init__(self):
        self.api_key = os.getenv("LANGCHAIN_API_KEY")
        self.client = None
        if self.api_key:
            self.client = Client(api_key=self.api_key)
    
    def validate(self) -> bool:
        """API 키 유효성 검증"""
        if not self.api_key:
            #print("API 키가 설정되지 않았습니다.")
            return False
        
        try:
            self.client.list_projects()
            #print("LangSmith API 키 유효성 검사 성공!")
            return True
        except Exception as e:
            #print(f"API 키 유효성 검사 실패: {str(e)}")
            return False
    
    def log_topic(self, language: str, difficulty: str, topic_data: dict) -> None:
        """토픽 정보 로깅"""
        if not self.client:
            return
            
        try:
            self.client.create_run(
                project_name="devita-1",
                name=f"{language}_{difficulty}_topic",
                inputs={
                    "language": language,
                    "topic": topic_data["topic"],           # 추가
                    "description": topic_data["description"], # 추가
                    "difficulty": difficulty
                },
                outputs={
                    "selected_topic": topic_data
                },
                run_type="retriever",
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            print(f"[{difficulty}] 토픽 로깅 성공")
        except Exception as e:
            print(f"토픽 로깅 실패: {str(e)}")

class MissionGeneratorChain(Chain):
    """미션 토픽을 검색하고 선택하는 체인"""
    
    client: Any = Field(exclude=True)
    collection_name: str = Field(description="ChromaDB collection name")
    difficulty_levels: List[str] = ["Advanced", "Intermediate", "Beginner"]
    
    def __init__(self, client: chromadb.Client, collection_name: str, **kwargs):
        super().__init__(**kwargs)
        self.client = client
        self.collection_name = collection_name
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def input_keys(self) -> List[str]:
        return ["language"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["missions"]

    def _retrieve_topics(self, language: str, difficulty: str) -> Dict:
        """특정 언어와 난이도의 토픽을 검색"""
        try:
            results = self.client.get_collection(self.collection_name).get(
                where={
                    "$and": [
                        {"language": {"$eq": language}},
                        {"difficulty_level": {"$eq": difficulty}}
                    ]
                }
            )
            if results["documents"]:
                random_idx = random.randint(0, len(results["documents"]) - 1)
                return {
                    "topic": results["metadatas"][random_idx]["topic"],
                    "description": results["metadatas"][random_idx]["description"],
                    "difficulty": difficulty
                }
        except Exception as e:
            print(f"토픽 검색 중 오류 발생: {e}")
        return None
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, List[Dict]]:
        """Chain 추상 메서드 구현"""
        return self.invoke(inputs)

    def invoke(self, inputs: Dict[str, str], return_only_outputs: bool = False) -> Dict[str, List[Dict]]:
        """각 난이도별 토픽 검색 실행"""
        missions = []
        for difficulty in self.difficulty_levels:
            topic_data = self._retrieve_topics(inputs["language"], difficulty)
            if topic_data:
                missions.append({
                    "topic": topic_data["topic"],
                    "description": topic_data["description"],
                    "difficulty": difficulty,
                })
        return {"missions": missions}

class MissionGenerator:
    """미션 생성을 총괄하는 메인 클래스"""
    
    def __init__(self, base_dir: str = "data_store"):
        load_dotenv()
        
        # 카테고리별 예시와 설명 정의
        self.category_examples = {
            "CS": {
                "description": "심화된 전공 수준의 개념 이해와 분석 능력을 테스트할 수 있는 미션",
                "subcategories": {
                    "DATA_STRUCTURE": {
                        "examples": """
                        높은 난이도: AVL 트리의 삽입과 삭제 과정을 구현하여 균형 상태를 유지하는 알고리즘을 작성하시오.
                        중간 난이도: 해시 충돌 해결을 위해 체이닝(Linked List)을 사용하는 해시 테이블을 구현하시오.
                        쉬운 난이도: 이진 탐색 트리에서 노드 삽입과 탐색 알고리즘을 구현하시오.
                        """
                    },
                    "ALGORITHM": {
                        "examples": """
                        높은 난이도: 다익스트라 알고리즘을 사용한 최단 경로 탐색 원리를 분석하고, 우선순위 큐를 이용한 최적화 방안을 설명하시오.
                        중간 난이도: 퀵 정렬 알고리즘의 분할 정복 방식을 설명하고, 시간 복잡도를 분석하며, 퀵 정렬의 성능을 개선하는 방법을 제시하시오.
                        쉬운 난이도: 선택 정렬 알고리즘의 동작 과정을 설명하고, 시간 복잡도와 공간 복잡도를 계산하시오.
                        """
                    },
                    "COMPUTER_ARCHITECTURE": {
                        "examples": """
                        높은 난이도: 캐시 메모리의 LRU(Least Recently Used) 정책을 하드웨어적으로 구현하는 알고리즘을 설계하시오.
                        중간 난이도: 파이프라인의 단계별 동작을 설명하고 간단한 명령어 흐름을 시뮬레이션하시오.
                        쉬운 난이도: 2진수 덧셈과 보수 연산을 사용하여 산술 연산을 수행하는 알고리즘을 구현하시오.
                        """
                    },
                    "NETWORK": {
                        "examples": """
                        높은 난이도: TCP와 UDP의 차이점을 설명하고, TCP의 혼잡 제어 메커니즘(예: 슬라이딩 윈도우, 혼잡 회피)을 분석하시오.
                        중간 난이도: 라우팅 테이블의 동작 원리와 OSPF 프로토콜의 작동 방식을 설명하고, 기본적인 OSPF 설정 예시를 작성하시오.
                        쉬운 난이도: OSI 7계층 모델의 각 계층 역할을 간단히 설명하고, 해당 계층에서 사용하는 프로토콜(예: HTTP, TCP, IP)을 예시로 드시오.
                        """
                    },
                    "DATABASE": {
                        "examples": """
                        높은 난이도: 병행 제어를 위해 다중 버전 동시성 제어(MVCC) 알고리즘을 구현하고 성능과 데이터 일관성 분석을 수행하시오.
                        중간 난이도: B+ 트리 인덱스를 이용해 효율적인 데이터 검색 알고리즘을 설계하시오.
                        쉬운 난이도: 관계 대수의 기본 연산을 사용하여 간단한 SQL 쿼리를 관계 대수로 변환하시오.
                        """
                    },
                    "OPERATING_SYSTEM": {
                        "examples": """
                        높은 난이도: 쓰레드 동기화 문제를 해결하기 위해 뮤텍스와 세마포어를 결합한 알고리즘을 구현하시오.
                        중간 난이도: 다단계 피드백 큐 스케줄링 알고리즘을 설계하고 시뮬레이션하시오.
                        쉬운 난이도: 페이징 기법에서 페이지 교체 알고리즘의 FIFO 방식을 구현하시오.
                        """
                    }
                }
            },
            "LANGUAGE": {
                "description": "프로그래밍 언어의 특성을 이해하고 활용하는 미션",
                "subcategories": {
                    "JAVA": {
                        "examples": """
                        높은 난이도: 멀티스레드 환경에서의 동기화 필요성을 설명하고, synchronized 키워드를 사용해 데이터 레이스 문제를 해결하는 방법을 분석하시오.
                        중간 난이도: 자바 메모리 구조(스택과 힙)의 차이점을 설명하고, 가비지 컬렉션 방식 중 '마크-스윕'의 작동 원리를 이해할 수 있도록 예시를 제시하시오.
                        쉬운 난이도: 객체 지향 프로그래밍의 4가지 개념(캡슐화, 상속, 다형성, 추상화)을 각각 설명하시오.
                        """
                    },
                    "JAVASCRIPT": {
                        "examples": """
                        높은 난이도: JavaScript의 프로토타입 체인을 활용하여 다중 상속 패턴을 구현하시오.
                        중간 난이도: JavaScript의 this 키워드가 다른 컨텍스트에서 어떻게 동작하는지 예제 코드를 작성하고 분석하시오.
                        쉬운 난이도: JavaScript에서 var, let, const의 차이를 설명하고 각 변수 선언 키워드의 예제 코드를 작성하시오.
                        """
                    },
                    "PYTHON": {
                        "examples": """
                        높은 난이도: Python의 메타클래스를 사용하여 커스텀 클래스 동작을 제어하는 로직을 구현하시오.
                        중간 난이도: Python의 제너레이터와 이터레이터의 차이를 설명하고 제너레이터를 이용한 데이터 스트림 생성기를 작성하시오.
                        쉬운 난이도: Python의 리스트 내포(List Comprehension)를 사용하여 1부터 100까지의 짝수 리스트를 생성하는 코드를 작성하시오.
                        """
                    },
                }
            },
            "TOOL": {
                "description": "개발 도구의 사용법과 핵심 기능을 실무적으로 활용하는 미션",
                "subcategories": {
                    "SPRING": {
                        "examples": """
                        높은 난이도: Spring과 JPA를 사용하여 다중 테이블 간 연관 관계를 매핑하고, 복잡한 쿼리를 최적화하는 방법을 설명하시오.
                        중간 난이도: Spring MVC 패턴을 활용하여 간단한 CRUD 기능을 갖춘 게시판 애플리케이션을 설계하고 구현하시오.
                        쉬운 난이도: Spring Boot를 사용하여 간단한 REST API를 생성하고, 이를 통해 기본적인 GET/POST 요청을 처리하시오.
                        """
                    },
                    "REACT": {
                        "examples": """
                        높은 난이도: React와 Redux를 사용하여 복잡한 상태 관리를 포함한 Todo 애플리케이션을 구현하시오.
                        중간 난이도: React에서 커스텀 Hook을 작성하여 데이터 페칭 로직을 재사용할 수 있도록 하시오.
                        쉬운 난이도: React 컴포넌트를 사용하여 단순한 버튼 클릭 카운터 애플리케이션을 작성하시오.
                        """
                    },
                    "PYTORCH": {
                        "examples": """
                        높은 난이도: PyTorch로 커스텀 신경망을 설계하고, 대규모 데이터셋을 활용하여 모델 학습 및 최적화 방법을 분석하시오.
                        중간 난이도: PyTorch에서 CNN을 사용해 이미지 분류 모델을 구현하고, 학습 및 평가 과정을 설명하시오.
                        쉬운 난이도: PyTorch에서 텐서 기본 연산과 자동 미분 기능을 사용하여 간단한 수학적 계산을 수행하시오.
                        """
                    },
                    "DOCKER": {
                        "examples": """
                        높은 난이도: Docker Compose를 사용하여 다중 컨테이너 기반의 마이크로서비스 아키텍처를 구축하시오.
                        중간 난이도: Dockerfile을 작성하여 Python 애플리케이션을 컨테이너화하고 이미지를 빌드 및 실행하시오.
                        쉬운 난이도: Docker CLI를 사용하여 간단한 Nginx 컨테이너를 실행하고 로컬에서 접근하시오.
                        """
                    }
                }
            }
        }
        
        self.db_dir = Path(base_dir) / "chromadb"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        try:
            self.collection = self.client.get_collection(
                name="mission_dataset",
                embedding_function=self.embedding_function
            )
        except InvalidCollectionException:
            self.collection = self.client.create_collection(
                name="mission_dataset",
                embedding_function=self.embedding_function
            )
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=1.0,          # 1.0 으로 조정
            frequency_penalty=0.7,    # 0.7 정도가 적당
            presence_penalty=0.7,     # 0.7 정도가 적당
        )
        
        self._setup_chains()

    def _setup_chains(self):
        """체인 설정"""
        self.retriever_chain = MissionGeneratorChain(
            client=self.client,
            collection_name="mission_dataset"
        )
        
        self.mission_prompt = ChatPromptTemplate.from_template("""
        당신은 개발자를 위한 미션 생성기입니다.
        주어진 토픽과 설명을 바탕으로 개발자의 실무 역량을 향상시킬 수 있는 미션을 생성해주세요.

        토픽: {topic}
        설명: {description}
        난이도: {difficulty}
        분야: {category_description}

        [행동 지침]
        1. 미션 제목만 출력합니다. 미션 설명, 미션 목표 등 추가 내용은 포함하지 않습니다. '높은 난이도:', '중간 난이도:', '쉬운 난이도:'와 같은 난이도 표시를 포함하지 마세요.
        2. 미션은 반드시 주어진 {topic}, {description}과 관련된 주제여야 하며, 그 외 내용은 포함하지 않습니다.
        3. 예시는 단지 참고용일 뿐이므로, 예시 내용에 너무 의존하거나 복사하지 마세요.
        4. 개념을 확인하고 이해를 테스트할 수 있는 미션을 중심으로 생성하세요.
        5. 미션 난이도는 높은 난이도, 중간 난이도, 쉬운 난이도 순으로 구분되며, 난이도 차이가 명확해야 합니다.
            * 높은 난이도(Advanced): 개념의 심화 응용이 필요한 수준 (아키텍처 설계, 성능 최적화 등)
            * 중간 난이도(Intermediate): 개념의 작동 원리를 이해하고 구현하는 수준
            * 낮은 난이도(Beginner): 기본 개념을 올바르게 사용할 수 있는 수준
        
        관련 예시:
        {category_examples}

        위 지침에 따라 한 줄의 미션 제목만 생성해주세요.
        """)
        
        self.generation_chain = LLMChain(
            llm=self.llm,
            prompt=self.mission_prompt
        )
    
    def get_category_from_topic(self, topic: str) -> str:
        """토픽에 해당하는 카테고리 반환"""
        topic = topic.upper()
        for category, topics in CATEGORIES.items():
            if topic in topics:
                return category
        return "CS"  # 기본값
    
    def generate_missions_from_db(self, topic: str) -> Dict:
        """ChromaDB 기반 미션 생성"""
        try:
            if topic in ["JAVA", "PYTHON", "JAVASSCRIPT"]:
                category = "LANGUAGE"
            elif topic in ["SPRING", "REACT", "DOCKER"]:
                category = "TOOL"
            else:
                category = "CS"

            topics_result = self.retriever_chain.invoke({"language": topic})
            if not topics_result or "missions" not in topics_result:
                raise ValueError("토픽 검색 결과가 없습니다.")

            generated_missions = []
            for mission_data in topics_result["missions"]:
                try:
                    result = self.generation_chain.invoke({
                        "topic": mission_data["topic"],
                        "description": mission_data["description"],
                        "difficulty": mission_data["difficulty"],
                        "category_description": self.category_examples[category]["description"],
                        "category_examples": self.category_examples[category]["subcategories"][topic.upper()]["examples"] 
                    })
                
                    generated_missions.append({
                        "difficulty": mission_data["difficulty"],
                        "mission": result["text"],
                        "topic": mission_data["topic"]
                    })
                except Exception as e:
                    print(f"개별 미션 생성 중 오류 발생: {e}")
                    continue

            return {"generated_missions": generated_missions}
        
        except Exception as e:
            print(f"미션 생성 중 오류 발생: {e}")
            return None
    
    def generate_missions_from_llm(self, topic: str) -> Dict:
        """LLM을 사용한 직접 미션 생성"""
        try:
            category = self.get_category_from_topic(topic)
            # 난이도별 미션을 저장할 딕셔너리
            difficulty_missions = {
                "Advanced": None,
                "Intermediate": None,
                "Beginner": None
            }
        
            # 각 난이도별로 미션 생성
            for difficulty in ["Advanced", "Intermediate", "Beginner"]:
                try:
                    result = self.generation_chain.invoke({
                        "topic": topic,
                        "description": f"{topic} 관련 미션 생성",
                        "difficulty": difficulty,
                        "category_description": self.category_examples[category]["description"],
                        "category_examples": self.category_examples[category]["subcategories"][topic]["examples"]
                    })
                
                    difficulty_missions[difficulty] = {
                        "difficulty": difficulty,
                        "mission": result["text"],
                        "topic": topic
                    }
                except Exception as e:
                    print(f"개별 미션 생성 중 오류 발생: {e}")
                continue
        
            # 순서대로 미션 리스트 생성
            generated_missions = []
            for difficulty in ["Advanced", "Intermediate", "Beginner"]:
                if difficulty_missions[difficulty]:
                    generated_missions.append(difficulty_missions[difficulty])
        
            return {"generated_missions": generated_missions}
            
        except Exception as e:
            print(f"미션 생성 중 오류 발생: {e}")
            return None
        
    def generate_missions(self, topic: str) -> Dict:
        """미션 생성 실행"""
        try:
            # ChromaDB 데이터가 있는 경우
            if topic in CHROMADB_TOPICS:
                return self.generate_missions_from_db(topic)
            # 그 외의 경우 LLM으로 직접 생성
            else:
                return self.generate_missions_from_llm(topic)
        except Exception as e:
            print(f"미션 생성 중 오류 발생: {e}")
            return None

def main():
    try:
        logger = LangSmithLogger()
        is_logging_enabled = logger.validate()
        
        mission_gen = MissionGenerator()
        
        #print("=== 미션 생성기 시작 ===")
        #print("지원하는 분야:")
        #for category, topics in CATEGORIES.items():
        #    print(f"\n{category}:")
        #    print("  " + "\n  ".join(topics))
        
        while True:
            topic = input("\n관심 분야를 입력하세요 (종료: q): ").strip().upper()
            if topic.lower() == 'q':
                break
            
            all_topics = []
            for topics in CATEGORIES.values():
                all_topics.extend(topics)
            
            if topic not in all_topics:
                print(f"지원하지 않는 분야입니다. 위 목록에서 선택해주세요.")
                continue
            
            print(f"\n{topic} 미션을 생성하는 중...")
            
            # 미션 생성 (json 형식)
            missions = mission_gen.generate_missions(topic)
            if missions and "generated_missions" in missions:
                generated_json = {
                    "missions": [
                        {
                            "difficulty": mission["difficulty"],
                            "mission": mission["mission"].strip(),
                            "topic": mission["topic"]
                        }
                        for mission in sorted(
                            missions["generated_missions"], 
                            key=lambda x: {"Advanced": 0, "Intermediate": 1, "Beginner": 2}[x["difficulty"]]
                        )
                    ]
                }
                print("\n=== 생성된 최종 미션 ===")
                print(json.dumps(generated_json, indent=2, ensure_ascii=False))
        
        print("\n미션 생성기를 종료합니다. 감사합니다!")
            
    except Exception as e:
        print(f"프로그램 실행 중 오류가 발생했습니다: {e}")
        
if __name__ == "__main__":
    load_dotenv(override=True)
    main()