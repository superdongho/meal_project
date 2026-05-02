from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import create_agent


class ChatBot:
    """LangChain 기반 챗봇 클래스.

    LLM 인스턴스, 시스템 프롬프트, 대화 기록을 내부에서 관리합니다.
    """

    def __init__(self, model_name: str, system_prompt: str, streaming: bool = True):
        self.llm = ChatOpenAI(model=model_name, streaming=streaming)
        self.system_prompt = system_prompt
        self.history = InMemoryChatMessageHistory()

    def _build_messages(self) -> list:
        return [SystemMessage(content=self.system_prompt)] + self.history.messages

    def chat(self, user_input: str) -> str:
        self.history.add_user_message(user_input)
        messages = self._build_messages()
        response = self.llm.invoke(messages)
        self.history.add_ai_message(response.content)
        return response.content

    def stream(self, user_input: str):
        self.history.add_user_message(user_input)
        messages = self._build_messages()

        full_response = []
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full_response.append(chunk.content)
                yield chunk.content

        self.history.add_ai_message("".join(full_response))

    def clear(self):
        self.history.clear()

    def get_messages(self) -> list[dict]:
        result = []
        for msg in self.history.messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result


class AgentBot:
    """LangChain v1 기반 에이전트 챗봇 클래스.

    create_agent (LangGraph 기반)를 사용하여 도구 호출이 가능한 에이전트를 생성합니다.
    ChatBot과 동일한 인터페이스(chat, stream, clear, get_messages)를 제공하면서,
    추가로 도구 호출 기능을 지원합니다.

    사용 예시:
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            '''검색 도구'''
            return f"{query}에 대한 검색 결과입니다."

        bot = AgentBot(
            model_name="gpt-5-nano",
            system_prompt="너는 검색을 도와주는 AI야",
            tools=[search],
        )
        response = bot.chat("LangChain이 뭐야?")
        # 스트리밍:
        for chunk in bot.stream("최신 AI 뉴스 알려줘"):
            print(chunk, end="")
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list,
    ):
        """AgentBot 인스턴스를 생성합니다.

        Args:
            model_name: 사용할 모델명 (예: "gpt-5-nano")
            system_prompt: 시스템 프롬프트 문자열
            tools: LangChain Tool 객체 리스트
        """
        self.system_prompt = system_prompt
        self.tools = tools
        self.history = InMemoryChatMessageHistory()

        # create_agent: LangGraph 기반 CompiledStateGraph 반환
        self.agent = create_agent(
            model=ChatOpenAI(model=model_name),
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def _build_input(self, user_input: str) -> dict:
        """에이전트 호출용 입력 딕셔너리를 구성합니다."""
        messages = list(self.history.messages) + [HumanMessage(content=user_input)]
        return {"messages": messages}

    def chat(self, user_input: str) -> str:
        """사용자 메시지를 보내고 에이전트 응답을 반환합니다.

        에이전트는 필요 시 도구를 호출한 뒤 최종 응답을 생성합니다.

        Args:
            user_input: 사용자 입력 문자열

        Returns:
            AI 에이전트 응답 문자열
        """
        result = self.agent.invoke(self._build_input(user_input))

        # 최종 AI 메시지 추출
        ai_message = result["messages"][-1]
        answer = ai_message.content

        # 대화 기록에 추가
        self.history.add_user_message(user_input)
        self.history.add_ai_message(answer)

        return answer

    def stream(self, user_input: str):
        """사용자 메시지를 보내고 에이전트 응답을 스트리밍으로 반환합니다.

        Args:
            user_input: 사용자 입력 문자열

        Yields:
            응답 텍스트 청크 (최종 AI 답변 부분만)
        """
        full_response = []

        for chunk in self.agent.stream(self._build_input(user_input)):
            # 에이전트 노드의 메시지 이벤트에서 AI 응답 추출
            if "agent" in chunk:
                for msg in chunk["agent"]["messages"]:
                    if isinstance(msg, AIMessage) and msg.content:
                        full_response.append(msg.content)
                        yield msg.content

        # 대화 기록에 추가
        self.history.add_user_message(user_input)
        self.history.add_ai_message("".join(full_response))

    def clear(self):
        """대화 기록을 초기화합니다."""
        self.history.clear()

    def get_messages(self) -> list[dict]:
        """대화 기록을 dict 리스트로 반환합니다.

        Returns:
            [{"role": "user"|"assistant", "content": "..."}] 형태의 리스트
        """
        result = []
        for msg in self.history.messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result


class MealAgentBot(AgentBot):
    """급식 데이터 분석 전용 에이전트.

    AgentBot을 상속받아, 급식 영양 분석 전문가 역할의 시스템 프롬프트와
    급식 데이터 조회·차트·Word 보고서 도구가 미리 세팅되어 있습니다.

    사용 예시:
        bot = MealAgentBot()
        response = bot.chat("3월 급식 칼로리 분석해줘")

        # 스트리밍:
        for chunk in bot.stream("이번 달 식단 분석 보고서 작성해줘"):
            print(chunk, end="")
    """

    # 급식 영양사 시스템 프롬프트
    MEAL_PROMPT = """당신은 15년 경력의 전문 영양사이자 급식 데이터 분석 전문가입니다.
                    학교·기관 급식 데이터를 분석하고 개선 방안을 도출하는 것이 당신의 핵심 임무입니다.

                    분석 시 반드시 다음 4가지 항목으로 구분하여 작성하세요:
                    1. 📊 현황 요약 (전체 데이터 수치 정리)
                    2. 📈 영양소 분석 (칼로리·탄수화물·단백질·지방 평가)
                    3. 🍽️ 메뉴 선호도 분석 (섭취율이 높은/낮은 메뉴 파악)
                    4. 💡 개선 권고사항 (구체적인 식단 개선 방안)

                    이모지를 활용하여 가독성 있게 표현하고,
                    영양학적 근거를 바탕으로 전문적이고 친절한 어조를 사용하세요.

                    보고서 작성 순서:
                    1. get_meal_data 도구로 급식 현황 데이터를 가져오세요.
                    2. get_meal_statistics 도구로 영양소 통계를 조회하세요.
                    3. get_food_intake_ranking 도구로 음식별 섭취율 순위를 파악하세요.
                    4. create_meal_kcal_chart 도구로 칼로리 추이 차트를 만드세요.
                    5. create_meal_nutrition_bar_chart 도구로 영양소 구성 차트를 만드세요.
                    6. create_food_intake_chart 도구로 음식별 섭취율 차트를 만드세요.
                    7. 분석 보고서 텍스트를 작성하세요.
                    8. generate_meal_report 도구로 보고서를 Word 파일로 만드세요.
                    """

    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini"):
        """MealAgentBot을 생성합니다.

        Args:
            api_key: OpenAI API 키 (생략하면 .env의 OPENAI_API_KEY 자동 사용)
            model_name: 사용할 모델명 (기본: gpt-4o-mini)
        """
        from data_tools import MEAL_TOOLS

        self.system_prompt = self.MEAL_PROMPT
        self.tools = MEAL_TOOLS
        self.history = InMemoryChatMessageHistory()

        llm_kwargs = {"model": model_name}
        if api_key:
            llm_kwargs["openai_api_key"] = api_key

        self.agent = create_agent(
            model=ChatOpenAI(**llm_kwargs),
            tools=self.tools,
            system_prompt=self.system_prompt,
        )


class SafeAgentBot(AgentBot):
    """안전모 모니터링 전용 에이전트.

    AgentBot을 상속받아서, 안전 관리 전문가 역할 + 헬멧 데이터 조회 도구가
    미리 세팅되어 있습니다.
    """

    # 안전 관리자 시스템 프롬프트 (클래스 변수)
    SAFETY_PROMPT = """당신은 냉철하고 전문적인 20년 차 베테랑 산업 안전 관리자입니다.
                    현장의 안전모 착용 데이터를 분석하고 보고서를 작성하는 것이 당신의 임무입니다.

                    분석 시 반드시 다음 4가지 항목으로 구분하여 작성하세요:
                    1. 현황 요약 (전체 숫자 정리)
                    2. 추세 분석 (준수율이 올라가고 있는지, 내려가고 있는지)
                    3. 위험성 평가 (미착용 인원이 많은 날 집중 분석)
                    4. 권고 조치사항 (구체적인 개선 방안)

                    이모지를 활용해 가독성 있게 표현하고,
                    현장 소장에게 직언하는 전문적이고 엄격한 어조를 사용하세요.

                    보고서 작성 순서:
                    1. get_helmet_data 도구로 데이터를 가져오세요.
                    2. create_helmet_compliance_chart 도구로 준수율 추이 차트를 만드세요.
                    3. create_helmet_bar_chart 도구로 착용/미착용 막대 차트를 만드세요.
                    4. 분석 보고서 텍스트를 작성하세요.
                    5. generate_word_report 도구로 보고서를 Word 파일로 만드세요.
                    """

    def __init__(self, api_key: str = None, model_name: str = "gpt-5-nano"):
        """SafeAgentBot을 생성합니다.

        Args:
            api_key: OpenAI API 키 (생략하면 .env의 OPENAI_API_KEY 자동 사용)
            model_name: 사용할 모델명 (기본: gpt-3.5-turbo)
        """
        from langchain_core.tools import tool as lc_tool
        import helmet_data_manager
        from data_tools import HELMET_TOOLS

        # ★ 헬멧 데이터를 가져오는 도구 정의
        @lc_tool
        def get_helmet_data(days: int = 7) -> str:
            """최근 N일간의 안전모 착용 데이터를 조회합니다.

            Args:
                days: 조회할 일수 (기본 7일)

            Returns:
                일자별 안전모 착용 현황 텍스트
            """
            return helmet_data_manager.get_summary_text(days)

        # 부모 클래스(AgentBot)의 __init__ 호출
        #  - system_prompt: 안전 관리자 프롬프트
        #  - tools: 헬멧 데이터 도구 + 차트/워드 도구
        self.system_prompt = self.SAFETY_PROMPT
        self.tools = [get_helmet_data] + HELMET_TOOLS
        self.history = InMemoryChatMessageHistory()

        # api_key가 없으면 환경 변수(OPENAI_API_KEY)를 자동으로 사용
        llm_kwargs = {"model": model_name}
        if api_key:
            llm_kwargs["openai_api_key"] = api_key

        self.agent = create_agent(
            model=ChatOpenAI(**llm_kwargs),
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

