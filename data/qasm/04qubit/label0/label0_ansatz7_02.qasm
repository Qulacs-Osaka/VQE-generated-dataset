OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.9197713683469431) q[0];
ry(1.503333824104888) q[1];
cx q[0],q[1];
ry(3.034824925691843) q[0];
ry(-1.7208625185450341) q[1];
cx q[0],q[1];
ry(0.15812010060385703) q[0];
ry(0.7115756062332306) q[2];
cx q[0],q[2];
ry(-0.6523116814544663) q[0];
ry(2.2691185758894377) q[2];
cx q[0],q[2];
ry(-1.9792995815166785) q[0];
ry(0.9259610433308781) q[3];
cx q[0],q[3];
ry(0.0727024796149126) q[0];
ry(1.835687502691288) q[3];
cx q[0],q[3];
ry(-1.163494281696078) q[1];
ry(1.1412421745616068) q[2];
cx q[1],q[2];
ry(1.3507549825642238) q[1];
ry(2.538917423075833) q[2];
cx q[1],q[2];
ry(-2.0028431830711515) q[1];
ry(0.6952990518249813) q[3];
cx q[1],q[3];
ry(1.1545545745600068) q[1];
ry(-2.4978043528016136) q[3];
cx q[1],q[3];
ry(-1.2047126388579392) q[2];
ry(2.396910521947579) q[3];
cx q[2],q[3];
ry(0.9833039113046342) q[2];
ry(1.4580978792078607) q[3];
cx q[2],q[3];
ry(1.6799371092250925) q[0];
ry(0.16071537004854794) q[1];
cx q[0],q[1];
ry(0.45990060526027143) q[0];
ry(-0.9518633413590489) q[1];
cx q[0],q[1];
ry(-3.0216542210982245) q[0];
ry(2.0849551978804595) q[2];
cx q[0],q[2];
ry(0.463404307454656) q[0];
ry(1.0670540524629581) q[2];
cx q[0],q[2];
ry(-1.9759272943405835) q[0];
ry(0.3478935593339534) q[3];
cx q[0],q[3];
ry(-0.8563091347334458) q[0];
ry(1.3372744782326578) q[3];
cx q[0],q[3];
ry(-0.47030368440639897) q[1];
ry(-3.0545002618080423) q[2];
cx q[1],q[2];
ry(-1.7962433844157633) q[1];
ry(-2.1831461010744775) q[2];
cx q[1],q[2];
ry(2.170459713487923) q[1];
ry(-1.797614228714891) q[3];
cx q[1],q[3];
ry(0.9093111554609957) q[1];
ry(-2.575553113087295) q[3];
cx q[1],q[3];
ry(-0.4976179776236081) q[2];
ry(0.7929181430752312) q[3];
cx q[2],q[3];
ry(0.315599428378288) q[2];
ry(-1.0852642216432677) q[3];
cx q[2],q[3];
ry(-1.3437914588031663) q[0];
ry(1.4403810493240699) q[1];
cx q[0],q[1];
ry(-0.672298492253275) q[0];
ry(2.4751491989683565) q[1];
cx q[0],q[1];
ry(-0.9065354369807508) q[0];
ry(-2.002547630898631) q[2];
cx q[0],q[2];
ry(-0.593833781677893) q[0];
ry(2.3866437150244044) q[2];
cx q[0],q[2];
ry(1.3129131148578888) q[0];
ry(-2.2143111984061674) q[3];
cx q[0],q[3];
ry(-2.299021432794431) q[0];
ry(2.158977195781624) q[3];
cx q[0],q[3];
ry(-2.247861653731105) q[1];
ry(0.8269389025457103) q[2];
cx q[1],q[2];
ry(2.709268462665718) q[1];
ry(-1.6932441717170663) q[2];
cx q[1],q[2];
ry(-0.13185021008512038) q[1];
ry(-1.6422681212212762) q[3];
cx q[1],q[3];
ry(-2.068083960876651) q[1];
ry(0.09647548316661858) q[3];
cx q[1],q[3];
ry(0.7452997411613662) q[2];
ry(-1.5384749436084892) q[3];
cx q[2],q[3];
ry(1.4108015905473956) q[2];
ry(-2.770979953076138) q[3];
cx q[2],q[3];
ry(0.43033479869033364) q[0];
ry(0.6458327920712702) q[1];
cx q[0],q[1];
ry(2.2012242153887245) q[0];
ry(2.025056310508266) q[1];
cx q[0],q[1];
ry(2.59175321667007) q[0];
ry(3.0875637537192198) q[2];
cx q[0],q[2];
ry(1.368260711748814) q[0];
ry(-1.3169486234765122) q[2];
cx q[0],q[2];
ry(-2.9475371552644685) q[0];
ry(2.7188147654795514) q[3];
cx q[0],q[3];
ry(0.39665055374394176) q[0];
ry(1.3060630331554257) q[3];
cx q[0],q[3];
ry(2.0448309477951976) q[1];
ry(1.5511441471059118) q[2];
cx q[1],q[2];
ry(-1.29713911781083) q[1];
ry(-0.4795156833524965) q[2];
cx q[1],q[2];
ry(-1.334656033406176) q[1];
ry(-1.998846909653171) q[3];
cx q[1],q[3];
ry(1.3651303090767855) q[1];
ry(-2.8060499921776527) q[3];
cx q[1],q[3];
ry(0.7280050006125112) q[2];
ry(-1.7587228695488821) q[3];
cx q[2],q[3];
ry(1.9722225063599237) q[2];
ry(-2.3606866468592465) q[3];
cx q[2],q[3];
ry(-0.33168319910966026) q[0];
ry(2.132435821049822) q[1];
cx q[0],q[1];
ry(1.1140330502569828) q[0];
ry(1.2812739411653533) q[1];
cx q[0],q[1];
ry(-2.061048219264233) q[0];
ry(-2.9388812485172107) q[2];
cx q[0],q[2];
ry(-1.1696797627517297) q[0];
ry(0.336882436824234) q[2];
cx q[0],q[2];
ry(1.7906507540342202) q[0];
ry(-2.9996086549948964) q[3];
cx q[0],q[3];
ry(-0.30174300483738936) q[0];
ry(2.5212372998102017) q[3];
cx q[0],q[3];
ry(2.2474315635078312) q[1];
ry(1.5643719030141503) q[2];
cx q[1],q[2];
ry(1.1328235159455282) q[1];
ry(-0.7632123301360263) q[2];
cx q[1],q[2];
ry(0.7313420410197099) q[1];
ry(-1.7906063899814582) q[3];
cx q[1],q[3];
ry(-1.7427612970371094) q[1];
ry(-0.1350737506123076) q[3];
cx q[1],q[3];
ry(2.9312727600112916) q[2];
ry(-2.2390143260124136) q[3];
cx q[2],q[3];
ry(0.023943558567218658) q[2];
ry(-2.4317144450359964) q[3];
cx q[2],q[3];
ry(-0.7687314851373585) q[0];
ry(1.3578896953898256) q[1];
ry(2.642823639959661) q[2];
ry(-3.0271534257463912) q[3];