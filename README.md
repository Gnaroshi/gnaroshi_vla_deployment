# gnaroshi_vla_deployment

로봇·모델별 프로젝트에서 가져다 쓰는 공통 GUI 배포 skeleton이다. 공통 저장소에는 GUI,
rollout 상태 관리, 영상·측정·수동 결과 기록, adapter 계약과 CPU mock만 둔다.
재사용 가능한 UR5e hardware adapter도 포함하며 특정 모델·논문·태스크의 실행 코드는 없다.

```text
src/gnaroshi_deployment/
  contracts.py             Observation / Prediction / Policy / Embodiment 계약
  session.py               worker, rollout 상태, 취소, 수동 outcome
  gui.py                   관측 화면, 버튼, 설정, notes와 결과
  records.py               JSON/JSONL, 영상, latency/실제 command Hz
  reference.py             호출 프로젝트가 제공하는 시작 장면 탐색
  desktop.py               현재 사용자의 desktop 환경 탐색
  adapters/mock.py         장치·모델 없이 실행 가능한 예제
  adapters/ur5e.py          모델과 독립적인 hardware adapter
scripts/deploy_mock.sh      CPU mock GUI launcher
tests/test_session.py       상태·취소·영상·hardware 계약 회귀 검증
```

## Mock 실행

Python 3.10 이상, NumPy, Pillow, Tk가 필요하다. 영상에는 OpenCV가 필요하다.
기존 호환 환경을 사용하거나 `pip install -e '.[video]'`로 설치한다. Tk는 Python/OS 패키지다.

```bash
DEPLOYMENT_PYTHON=python3 \
DEPLOYMENT_OUTPUT_ROOT=/absolute/run-storage/mock \
  bash scripts/deploy_mock.sh
```

기본은 현재 DISPLAY의 mock GUI이며 모델, 정책 서버, 로봇, 카메라에 연결하지 않는다.
`DEPLOYMENT_OUTPUT_ROOT`를 생략하면 이 checkout의 ignored `.runtime/mock`에 기록한다.
Python CLI도 같은 환경변수 또는 `--output-root`로 경로를 지정하며 기본은 `.runtime/results`다.
SSH에서 GUI가 필요하면 대상 desktop 또는 trusted X11 forwarding을 사용한다.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m gnaroshi_deployment.cli \
  --output-root /absolute/run-storage/mock --headless-demo
```

`--headless-demo`는 mock rollout을 자동 실행해 outcome 저장·home 복귀를 검증한다.
실제 모델 factory와 함께 사용할 수 없다. 가상 display에서는 `--gui-smoke`와
`--screenshot /absolute/run-storage/gui.png`로 GUI 경로를 확인할 수 있다.

## 프로젝트에서 확장

공통 저장소는 skeleton으로 유지한다. 각 연구 프로젝트 안에 clone한 작업 사본에서
`<backbone>/<paper>/` 아래에 전용 코드를 관리한다. 실행 경로에 모델·논문 이름을 드러낸다.

```text
<project>/gnaroshi_vla_deployment/
  src/gnaroshi_deployment/       공통 skeleton
  <backbone>/<paper>/
    deploy.sh                   task/model/checkpoint/control 설정
    launch.py                   해당 정책 프로세스와 GUI의 lifecycle
    policy.py                   모델 전처리·추론·reset·latency adapter
    tasks.json                  해당 프로젝트의 instruction/home/장치 설정
```

1. `contracts.py`와 `adapters/mock.py`를 읽고 `(SessionConfig, Policy, Embodiment)`를 반환하는
   factory를 구현한다. `python -m gnaroshi_deployment.cli --factory MODULE:FUNCTION --config FILE`로 연결한다.
2. image role/RGB·resize, state/action 차원·좌표·단위·정규화, horizon과 실제 실행 길이,
   reset 및 checkpoint binding을 해당 모델의 실제 구현에서 확인한다. 다른 모델에서 추정하지 않는다.
3. 새 로봇은 `connect/observe/home/begin_episode/apply/stop/close` 계약을 구현한다.
   hardware adapter에는 모델 코드를 import하지 않는다. 카메라 역할은 `SessionConfig.camera_roles`로 지정한다.
4. launcher 상단에서 task, method, checkpoint, target Hz, max steps, camera와 output을 수정하게 한다.
   모델 이름·논문 이름·실행 mode가 GUI와 저장된 config에도 나타나야 한다.
5. 모델·runtime·checkpoint·cache는 소유 프로젝트에서 관리한다. 장치 주소, camera serial,
   task home과 로봇별 규약도 프로젝트가 공급한다. UR5e adapter는 이 값이 없으면 생성을 거절한다.
6. 현재 할당받은 GPU를 각 실행에서 명시한다. 컨테이너에 보이는 장치와 실제 사용 배정은 구분한다.
7. import/inspect → CPU tests → mock GUI → 장치 미연결 정책 preflight → operator 실험 순서로 검증한다.

프로젝트별 변경을 공통 skeleton의 기본 branch에 섞지 않는다. 공통 기능 개선만 분리해서 반영한다.

## GUI 동작

| 동작 | 결과 |
|---|---|
| Start `[N]` | task home → policy reset → fresh rollout |
| Success `[S]`, Failure `[F]` | 명령 중단 → 수동 결과/영상 저장 → task home |
| Stop `[X]`, Retry `[R]` | 현재 attempt 폐기 → 현 위치 유지 |
| Save `[W]` | 결과와 notes 저장 |
| Delete last `[D]` | 확인 후 마지막 저장 결과/영상 삭제 |
| Exit `[Esc]` | pending action 취소 → 미분류 attempt 폐기 → 장치 종료 |

`max_steps`에 도달하면 정지 후 operator outcome을 기다린다. 자동 성공/실패로 분류하지 않는다.
target Hz와 max steps는 READY에서 수정할 수 있다. 지연된 추론 결과는 Stop 이후 실행되지 않는다.
성공/실패 뒤 home의 완료·취소·오류도 기록한다. 통신/장치 오류는 수동 task 실패와 구분한다.

### 같은 창에서 checkpoint 교체

프로젝트 launcher는 선택 항목과 새 세션 factory를 GUI에 전달할 수 있다.

```python
DeploymentGUI(root, session,
    reload_fields=[
        {"key": "checkpoint", "label": "Checkpoint", "kind": "directory"},
        {"key": "mode", "label": "Mode", "kind": "choice", "choices": ["default", "alternative"],
         "value": "default"},
    ],
    reload_factory=make_session)
# make_session(selection: dict, previous_config: SessionConfig) -> unstarted DeploymentSession
```

`kind`는 `directory`, `file`, `text`, `choice`를 지원한다. 초기 선택은
`config.metadata['reload_selection']`이 우선하며 각 field의 `value`가 기본값이다.
선택 패널은 기본으로 접혀 있으며 헤더의 `Change checkpoint…`로 펼친다. Load 요청을 수락하면
다시 접어 결과표·notes 공간을 확보한다. 로드 오류는 상태 표시줄에 남고 패널을 다시 열어 수정한다.
factory는 선택한 checkpoint/method를 config와 policy metadata에 명시하고, 필요한 경우
checkpoint별 `output_root`를 정한다. factory에서 세션을 `start()`하지 않는다.
factory가 정규화한 `reload_selection`은 그대로 보존하고 입력한 값은 `reload_request`에 기록한다.
재로딩 후 선택 항목도 정규화한 경로와 설정으로 갱신한다.
factory가 실패하면 자체 생성한 서버·프로세스도 정리한 뒤 예외를 반환해야 한다.

Load checkpoint는 READY, 정상 종료된 세션 또는 정책 로드 단계에서 안전하게 종료된 ERROR에서
활성화된다. 실행·home·stop 중에는 교체할 수 없다. GUI는 notes를 저장하고 기존 세션의
worker 종료·장치 정지 확인·cleanup을 기다린 뒤 background thread에서 factory를 호출한다.
새 세션은 같은 task와 camera roles를 유지하며 별도 결과 폴더를 사용한다. 창과 선택 항목은
유지하고 결과표·notes·측정·폴더 표시는 새 세션으로 바꾼다. 이전 결과·영상은 보존한다.

잘못된 경로나 정책 로드 실패는 선택을 수정해 재시도할 수 있다. 추론·장치 오류, 정지 미확인,
cleanup 실패는 교체를 차단한다. 로딩 중 Exit는 factory가 반환한 세션을 연결하지 않고 정리한 뒤
창을 닫는다. 상태 검사는 `session.can_replace`, 교체 직전 종료 확인은
`session.replacement_closed`, 종료 대기는 `session.shutdown_complete`로 구분한다.

UR5e 연결 단계는 gripper 상태를 읽으며 activation은 operator Start 이후에 수행한다.
activation 자체가 손가락 움직임을 만들 수 있다. 별도 Python open–close calibration 루프는 없다.
하드웨어 adapter의 실제 수행 능력은 CPU mock 검증과 별개다.

## 기록과 영상

```text
manifest.json                   config, source 상태, policy/robot/camera metadata
events.jsonl                    operator 요청, 폐기/삭제, 오류
results.json / notes.txt        수동 결과·성공률과 notes
rollout_0001_.../
  result.json                   outcome, home 상태, 실제 Hz, latency 통계
  steps.jsonl                   state/action/command/그리퍼와 단계별 시간
  <camera-role>.mp4            실제 capture 시간에 대응하는 영상
  frames.jsonl                  source 관측 step과 video frame 매핑
```

actual command Hz는 `(commands-1)/(last_command_time-first_command_time)`로 계산한다.
policy/client/RPC/server/GPU 지연은 제공된 실제 측정값만 구분해서 집계한다.
Capture, command, 영상 encoding 시간과 deadline miss도 따로 저장한다.

영상은 policy가 본 관측을 monotonic capture 시간으로 30fps에 재표본화한다. 새 관측 전까지
이전 이미지를 유지하고 마지막 pending 이미지는 종료 시 기록한다. target Hz로 영상 fps를
가정하지 않는다. 독립 observer camera의 연속 녹화는 별도 adapter가 필요한 확장이다.

시작 장면 browser는 호출 프로젝트의 `reference_manifest`를 사용한다. manifest는
`task/instruction/episode_count`와 step0 sample의 primary/wrist 파일 및 optional checksum을 포함한다.
다른 task, 경로 탈출과 checksum 불일치를 거절하며 reference 이미지는 policy 입력이 아니다.
