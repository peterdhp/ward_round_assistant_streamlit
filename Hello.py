import streamlit as st 
import requests
import uuid
import time
import json
import os
import io
from io import BytesIO
from audiorecorder import audiorecorder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI


from openai import OpenAI


openai_api_key = st.secrets['OPENAI_API_KEY']
CLOVA_INVOKE_URL = st.secrets['CLOVA_INVOKE_URL']
CLOVA_SECRET_KEY = st.secrets['CLOVA_SECRET_KEY']

def strip_to_start_with_byungdong(s):
    start_index = s.find('병동')
    if start_index == -1:
        return ''
    return s[start_index:]
    
def patient_list_extractor(image) :
    #NAVER CLOVA OCR image
    
    
    api_url =CLOVA_INVOKE_URL
    secret_key =CLOVA_SECRET_KEY
    #image_file = '환자리스트.png'

    request_json = {
        'images': [
            {
                'format': 'png',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    #files = [('file', open(image_file,'rb'))]
    files = [('file',BytesIO(image.getvalue()))]
    
    headers = {
    'X-OCR-SECRET': secret_key
    }

    response = requests.request("POST", api_url, headers=headers, data = payload, files = files)
    result = response.json()
    text = ""
    for field in result['images'][0]['fields']:
        text += ' '+ field['inferText']
        if field['lineBreak']:
            text += '\n'
            
    text = strip_to_start_with_byungdong(text)
            
    return text

def ward_round_summary_with_list(model):
    llm = ChatOpenAI(model=model, temperature=0) 
    
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are medical student shadowing a ward round of the orthopedic department. 
Given the transcript of the ward round and the patient list table, write a SOAP progress note. 
ONLY include patients in the transcript given. Unless for diagnostic/medical terms use Korean

An example of what the summary of each patient should be :
환자 :  홍길동 님 #if you can not identify the patient, leave it blank
S(Subjective information) : 통증이 호전됐다고 함 
O(Objective information): improved right wrist ROM flextion[30--> 90]
A(Assesment) : PEN 후 postoperative care #Use medical terms in english
P(Plan) : 진통제 용량 감소

notes : #other important conversations that are worth note-taking

patient list table:
{patient_list}

""",
            ),
            ("human","{transcripts}" ),
        ]
    )
    
    chain = prompt | llm | output_parser
    #answer = chain.invoke({'medical_record' :medical_record})
    
    
    return chain



class NamedBytesIO(io.BytesIO):
    def __init__(self, buffer=None, name=None):
        super().__init__(buffer)
        self.name = name

if 'OCR_status' not in st.session_state:
    st.session_state.OCR_status = False
if 'transcript_status' not in st.session_state: 
    st.session_state.transcript_status =False
    
if 'transcript' not in st.session_state:
    st.session_state.transcript =''



st.title('회진 도우미')

#회진 리스트 업로드하기
patient_list_img = st.file_uploader('회진 리스트 사진을 업로드 해주세요', type=None)



if patient_list_img : 
    st.image(patient_list_img)
    patient_list_str = patient_list_extractor(patient_list_img)
    st.session_state.OCR_status = True
    #st.write(patient_list_str)

#음성녹음 파일 올리기
st.text('회진을 시작하면서 녹음을 시작해주세요.')
st.session_state.audio=audiorecorder(start_prompt="", stop_prompt="", pause_prompt="", key='recordings')
thirty_minutes = 30 * 60 * 1000
if len(st.session_state.audio)>thirty_minutes:
    st.warning('음성 녹음은 30분을 초과할 수 없습니다. 첫 30분에 대한 진료내용만 사용합니다.', icon='⚠')
    st.session_state.audio = st.session_state.audio[:thirty_minutes]

if st.session_state.recordings and len(st.session_state.audio)>100 :
    player_field = st.audio(st.session_state.audio.export().read()) 

#음성파일 whisper 돌리기  
    if not st.session_state.transcript_status :
        client = OpenAI(api_key=openai_api_key)
        with st.spinner('음성 녹음을 받아적고 있습니다...'):
            asr_result = client.audio.transcriptions.create(model="whisper-1", language= "ko",file= NamedBytesIO(st.session_state.audio.export().read(), name="audio.wav"))
        st.session_state.transcript += '\n'+ asr_result.text 
#         st.session_state.transcript ="""안녕하세요 최명희님, 허리 그리고 다리 쪽 아프신 거 어떠세요? 다리는 좀 안 좋고요, 허리는 괜찮아요. 지난번에 처음 오셨을 때 넘어지시고 나서 다리 많이 아프다 하셨잖아요. 그게 좀 좋아지셨어요? 그냥 잘 모르겠어요. 계속 불편하시긴 해요. 손 한번 보실까요? 손 이렇게 쥐었다 폈다 해보세요. 왼손도 같이 쥐었다 폈다 해보세요. 왼손 펴보세요. 이쪽이 좀 안 좋아요. 이렇게 안 되신 지는 얼마나 되셨어요? 이렇게 안 되네요? 네, 왼손 이렇게 쭉 펴보세요. 다치고 나서 이렇게 된 거예요. 넘어지시고 나서? 원래는 왼손 잘 쓰셨어요? 그럼요. 지금 오른손으로 젓가락질 잘 하세요? 네, 잘해요. 원래는 잘 됐는데 이게 안 돼요? 여기 넘어졌거든요. 네, 그래가지고 쭉 펴보고 이런데 만지면 느낌 있어요 여기? 아프죠? 아픈가요? 팔꿈치 구부렸다 폈다 해보세요. 어깨도 이렇게 움직여 보시고. MRI는 어디 뭐 있어요? 저희 이따가 아드님 오시면 같이 설명 드릴게요. 좀 설명 드릴 내용들이 꽤 있어서 직접 보고 아무래도 보호자분께서 같이 들으시는 게 좋을 것 같습니다. 이따 뵐게요. 연락 드리겠습니다.
# 안녕하세요. 채복순님. 잘 주무셨어요? 별일 없으셨죠? 여기 한번 두드려 볼게요. 통증 정도 어때요? 엄청아프지는 않아요. 많이 좋아졌어요? 여기는? 너무 아프시고? 그래요, 알겠습니다. 다른 문제는 없으시죠? 머리 아프거나 어지럽고 이런 거는? 네, 알겠습니다.
# 안녕하세요. 어때요 다리? 화요일날 주사 맞고 통증을 가라앉혔잖아요. 근데 이 약이 진통제가 덜 되나 봐. 어저께는 좀 설쳤어. 잠을 아파서. 어디가 아프셨어요? 다리가 아파서. 진통제가 약한가? 아니요, 진통제는 충분히 세게 드리고 있는데요.어저께 화요일날 주사 맞았어. 굉장히 편해서 잠을 잤거든. 근데 어저께는 그 정도보다는 약하지만 설쳤어. 그래서 지금 잠을 자야 될 것 같은데. 어제 낮에 아프셨어요? 어저께 낮에는 넘어갔는데 저녁에 이렇게… 낮에는 그래도 지낼만 하셨어요? 낮에는 지나갔어. 낮에는 지낼만 하셨다는 거죠? 많이 불편하지 않고? 저녁에는 아무나 저녁에는 그쵸. 그러니까 낮에는 괜찮으셨다는 거죠? 겨우 이렇게 견딜만하게. 견딜만한 정도? 그럼 저녁약을 조금 더 올려드릴게요. 저녁약이 따로 있어요? 저녁에 드시는 이거 다 맞춤 먹는 거에요? 저녁에 드시는 건 다 올려드릴게요. 그게 따로 있어요? 저녁에 먹는 것도 먹으면 낮에도 다 똑같지 않아요. 아픈데 진정이 되는 게 똑같아. 근데 약을 무작정 세게 쓴다고 다 좋은 것도 아니니까 일단 저녁약만 좀 조절해드릴게요. 네, 그러면 저녁약 잠을 못 주무셨다니까 이따가 궁뎅이 주사 가서 맞으면. 네, 그렇게 하셔도 됩니다. 궁뎅이 주사 맞으면 괜찮을까요? 궁뎅이 주사는 몇 시간 가요? 한 7-8시간 생각하시면 돼요. 그런가봐. 화요일날 맞았을 때는 편안했어. 밤에 잠을 자고. 근데 어저께는 잠을 좀 설치고 그 정도로 심하진 않지만 상처싸고 이전보다 이전만큼 심한 건 아니라는 거죠? 화요일보다 심하진 않은데 좀 설쳤어. 지난주보다는 지금 많이 좋아졌죠? 그렇지, 지난주 처음 오셨을 때보다는 그럼 그쵸? 그쵸, 그쵸. 이 주사를 한 대 맞을까 지금. 네, 알겠습니다. 네, 그렇게 해드릴게요. 감기가 약해서 진통제가 약해서 건강이 잘못될 만큼 높였다 그치? 네네, 알겠습니다. 이거 갖고 이제 해결을 해야 되는 거 아니에요? 제가 이 약 먹으면서. 네, 조금 더 올려드릴게요. 조금 더 알겠습니다.
# 안녕하세요. 별일 없으셨죠? 오늘 예정대로 퇴원하시고 고생 많으셨습니다 그동안. 네, 감사합니다. 지금 여기 조금 먹먹한 거는 아마 있으실 거고 무릎쪽에. 그다음에 통증은 없으시죠? 네, 좋아요. 관리 잘 하시고. 네, 제가 다음에 또 외래에서 뵐게요. 네, 감사합니다. 네, 수고하셨습니다.
# 안녕하세요. 안계시네. 오시네요. 안녕하세요. 오늘 집에 가시죠. 네, 집에 가서도 넘어지지 않게 조심하시고 아마 통증은 점점 더 좋아지실 거예요. 네, 고생 많으셨고 나중에 외래에서 뵙겠습니다.
# 안녕하세요. 좀 어떠세요 허리? 통증은 좀 있어서. 괜찮다 안 괜찮다 하세요? 우리 피부 한번 볼까요? 네, 괜찮습니다. 처음보다. 그럼 우리 월요일보다 어때요? 월요일보다 더 나아요? 네, 점점 시간이 갈수록 좋아지네요. 그래도 오늘 주사 한번 더 맞아볼까요? 네, 알겠습니다. 아직 진통제 좀 필요할 것 같아요. 네, 네 알겠습니다. 그럼 주사 맞을게요. 오후에 뵐까요? 잠깐 외래에서 시간이 될 때 연락드릴게요. 
# 이거 이렇게 하지 말고 펴봐요. 너무 아파가지고. 할 수 있어요 왜 못해요 지금 하시잖아 지금 아 하는데 지금 앉아서 얼마나 좋아요? 지금 이렇게 서계시니까 얼마나 보기 좋아요? 아니 또 앞에 짚는다 또 얼마나 보기 좋아요? 이렇게 앉고 계시면 아이고 나도 그런거 싫어요 볼까요? 저거 그거 그거 뭐야 방광염 같이 어저께서부터 오줌도 못하고 그냥 자꾸 옷에다가 그리고 여기가 아주 막 소변 참아서 그런 것처럼 소변이 시원하게 나오지도 않으면서 찔끈찔끈찔끈 그래요 그리고 아 이놈의 열감 좀 나 혼자 느끼는 열감 좀 없었으면 좋겠어요 일단 비뇨의학과 진료 보시게 해드리고요 fever 없으시죠? 네 fever 없으시죠? 선생님 내분지 내과도 있나요? 있어요 아 그러면 그쪽에서 진료를 같이 좀 보라고 하시겠어요? 어떤거 때문에요? 열나는거 때문에 예전에도 그쪽으로 한번 검사를 받아보라고. 갑상선? 갑상선은 지금 혹 때문에 정기검진을 계속 하시고 계시긴 하시거든요. 예전에 어떤 이유로 내분비내과 보라고 들으셨어요? 이 열나는거 때문에 예전에도 몇년전에도 한번 그러셔가지고 신장검사부터 뭐 염증수치까지 다 검사했는데 결국엔 안되니까 내분비내과로 의뢰를 해주셨었는데. 어디 병원에서요? 저희 다니는 병원에서 대학병원에서 그때 열감이 조금 잦아들면서 괜찮아지셔서 거기까지는 저희가 진행을 하셨습니다. 갑상선 때문에 그런건가? 갑상선이 혹만 있고 뭐 약 먹거나 그런 약은 여태 한번도 안먹어봤는데 혹시라도 지금 계속 이 열때문에. 갑상선 포함으로 내분비내과 진료 원하셔서 아니 저거 지금 열만 덜나도 . 그런데요 예전에도 그러신적이 또 있으시죠? 옛날에 한번 한번 그러셔가지고 검사 전체적으로 염증검사 다 하셨었거든요. 이상은 없다고 그랬는데. 알겠습니다. 그러면 오늘 비뇨의학과랑 내분비내과 보시게 해드리고 어제 그저께 찍은 엑스레이는 괜찮으세요. 소변검사는 이번에 소변검사도 특별한건 없었어요 혈액검사도 혈액검사의 주 목적은 수술하시고 나서 염증수치가 오르는데 원래 정상적으로도 올라요. 그런데 그게 잘 떨어지는걸 보는거고 많이 떨어지셨어요 그래서 뭐 큰 문제는 없으시고 저는 지금 열감 나는 것만 없어져도. 알겠습니다. 걷는건 얼마나 걸으시죠? 걷는거는 오늘 아침에도 5시에 일어나서 6시까지 안 누웠었어요. 여기 방금 좀 걷고. 어제도 계속. 아 네 좋아요. 넘어지지 않게 조심하시고. 열때문에? 거기다가 또 오한이 나요 추워가지고 알겠습니다. 저희가 한번 의뢰드려 볼게요 네."""
        st.session_state.transcript_status =True
if st.session_state.transcript_status and st.session_state.OCR_status :
    SOAPnote = ward_round_summary_with_list('gpt-4o').stream({'transcripts': st.session_state.transcript, 'patient_list' :patient_list_str})
    st.write_stream(SOAPnote)
    #st.text(SOAPnote,flush=True)
#output