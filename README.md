# nariflow
 
[레퍼런스][https://www.hanbit.co.kr/store/books/look.php?p_code=B6627606922]를 기반으로 핵심 기능을 구현했습니다(Parameter, Variable, Function)

다만, nariflow는 다음의 점에서 레퍼런스에서 구현한 딥러닝 프레임워크와는 차별점이 있습니다.

1) 레퍼런스에선 다루지 않지만 텐서플로우에서 활용하고 있는 GradientTape를 도입하여 더 효율적인 처리가 가능하도록 했습니다.
2) 레퍼런스에선 다루지 않는 야코비 행렬 출력 / 헤쎄 행렬 계산 기능을 추가했습니다.

#Installtion

pip install git+https://github.com/dlanjrjs/nariflow.git

#작동 구조 설명

![Alt text](/images/슬라이드2.png)
![Alt text](/images/슬라이드3.png)
![Alt text](/images/슬라이드4.png)
![Alt text](/images/슬라이드5.png)