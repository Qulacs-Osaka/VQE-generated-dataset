OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.04667858421915615) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.01719041672358931) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09921943415235462) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.13628781048014968) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.0841428913578213) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.060293898108161416) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.06225435668106426) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.012613786353924501) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06558677319561897) q[3];
cx q[2],q[3];
rz(-0.10294114463588974) q[0];
rz(-0.07728261458039927) q[1];
rz(-0.06727706813925724) q[2];
rz(-0.08146882117255998) q[3];
rx(-0.20950409150479743) q[0];
rx(-0.08018530162429985) q[1];
rx(0.03132004749165408) q[2];
rx(-0.2745857695072166) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03640764814590963) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0023634470815230607) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.008713945645147223) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.028258805757265883) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.034701171806894485) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.12078234950146864) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07509532910944336) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.023924542668868495) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03283353528868153) q[3];
cx q[2],q[3];
rz(-0.07042686584610978) q[0];
rz(-0.0024517366238845815) q[1];
rz(-0.0393676871088139) q[2];
rz(-0.07543429730446286) q[3];
rx(-0.22804757633594513) q[0];
rx(-0.08724690225534698) q[1];
rx(0.06455960705475224) q[2];
rx(-0.24672092244747548) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07913911785517701) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.010639034959095834) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.041441453102191814) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.017112487190381276) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.030820838745732042) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.08231246383504244) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.12288063050081942) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.03415254250166383) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03868808427824519) q[3];
cx q[2],q[3];
rz(-0.10228316294478595) q[0];
rz(-0.05839892147033924) q[1];
rz(-0.09148244368030942) q[2];
rz(-0.1079646725120661) q[3];
rx(-0.20607340696779558) q[0];
rx(-0.1617980042864607) q[1];
rx(-0.032588644070436594) q[2];
rx(-0.31035538900958565) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0804567685100991) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0616742346145894) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.03856069580400278) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.09396714511715404) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07669896658856802) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.08370583464989005) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.11700271753419994) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.050424250230509865) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.022533185045424294) q[3];
cx q[2],q[3];
rz(-0.09715776061909606) q[0];
rz(-0.03828258816631416) q[1];
rz(-0.051055655552644526) q[2];
rz(-0.08926802765630985) q[3];
rx(-0.2550105826819848) q[0];
rx(-0.11601868889829774) q[1];
rx(-0.07471958503444787) q[2];
rx(-0.2432033462047658) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.09987035902388418) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10285845261838468) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0021220244120469184) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.13409277951967324) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.15725551417355568) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.04005105332236435) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.14059584845840065) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.06300256768517201) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.028062366108670706) q[3];
cx q[2],q[3];
rz(-0.0816608340310827) q[0];
rz(-0.026407620721556502) q[1];
rz(0.10836693456771876) q[2];
rz(-0.15383230249425367) q[3];
rx(-0.24874033921956926) q[0];
rx(-0.1340551538749021) q[1];
rx(-0.04659913095821228) q[2];
rx(-0.299576617873746) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.09304301209905642) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13210488992700922) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.06865861295274328) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.06912809663010715) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.09737204880681463) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.08781443793002972) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1276320842423431) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.09050996293969481) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07052575857404988) q[3];
cx q[2],q[3];
rz(-0.02796156001649605) q[0];
rz(0.029865228819542056) q[1];
rz(0.1335992670976923) q[2];
rz(-0.1204359386595924) q[3];
rx(-0.19009205157320525) q[0];
rx(-0.11519458085292465) q[1];
rx(-0.06456980276061651) q[2];
rx(-0.24001951432332133) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.05694444660197325) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.05651797588540308) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05927883014579616) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.15219041949027085) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.16771584279293858) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.007827405755629806) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05950224344367044) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.10320389416032755) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.10026940456969594) q[3];
cx q[2],q[3];
rz(-0.09303387132594687) q[0];
rz(0.04580239211947053) q[1];
rz(0.136778293940436) q[2];
rz(-0.17533680507684024) q[3];
rx(-0.1997012775232705) q[0];
rx(-0.0893071420164304) q[1];
rx(-0.057608393168411653) q[2];
rx(-0.3124194189790849) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.15423114468774937) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07621370619264405) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04319423980729065) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.16535111980782824) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.04016522208878795) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11466536373111731) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09700867930442189) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.162809052628753) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03265501940706285) q[3];
cx q[2],q[3];
rz(-0.04946812770907763) q[0];
rz(-0.023583615808091665) q[1];
rz(0.15584744996118996) q[2];
rz(-0.1778153650636925) q[3];
rx(-0.24052375063758102) q[0];
rx(-0.09104908704675822) q[1];
rx(-0.18275464188616064) q[2];
rx(-0.22904131453604473) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.09240820755068811) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.05637070819578828) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.02220648591361835) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.13438743385193333) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.055152063164335526) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06562128435330082) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.06409955933754237) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.052845121580007655) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0511813820726901) q[3];
cx q[2],q[3];
rz(-0.06779368049871536) q[0];
rz(0.029022793187653263) q[1];
rz(0.1790033957033645) q[2];
rz(-0.1577748786318853) q[3];
rx(-0.2350472510469802) q[0];
rx(-0.0828077897853603) q[1];
rx(-0.22022634761166562) q[2];
rx(-0.1878662710356846) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07517532883499466) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.15569602050319836) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.027649942714736978) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.06183593199722561) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.04351923739272404) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.028252828300240812) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.034087952911399766) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.10734550757474154) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.08139421777406644) q[3];
cx q[2],q[3];
rz(0.006366392906302108) q[0];
rz(0.03871941860997699) q[1];
rz(0.15936550759154589) q[2];
rz(-0.09564090522258979) q[3];
rx(-0.14467491898423682) q[0];
rx(-0.12967237841072857) q[1];
rx(-0.26572609929291785) q[2];
rx(-0.15486517568565114) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.02889977230220163) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.08662092991649047) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.008104370332299679) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.015190282524479912) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.05123681125560058) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.04840736548616649) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0794124606852674) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.08340803285389789) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.14692842148488533) q[3];
cx q[2],q[3];
rz(0.06373079936675294) q[0];
rz(0.07164061672190529) q[1];
rz(0.21496308215006152) q[2];
rz(-0.052702327107810136) q[3];
rx(-0.13322313185175988) q[0];
rx(-0.07666200723292271) q[1];
rx(-0.2805605785398345) q[2];
rx(-0.19088214342020254) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.04129550917488554) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09733817946483184) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03454461158232551) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.05504837548915845) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.026918329402128613) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.09899152162480972) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.055107268738055284) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07294580370584873) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1451454009280024) q[3];
cx q[2],q[3];
rz(-0.0033772389093359007) q[0];
rz(0.08367117187570626) q[1];
rz(0.11393824294007483) q[2];
rz(0.06491279937268804) q[3];
rx(-0.19014114266606535) q[0];
rx(-0.12918636925163954) q[1];
rx(-0.2164591001373333) q[2];
rx(-0.16613380324495747) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03259488474476221) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.16308724209909659) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.00989160967414988) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.15345932940407278) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.01967177987124775) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.03156323418126895) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.023945351983084468) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.14238909336273486) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.050040534768090124) q[3];
cx q[2],q[3];
rz(0.11475683593833) q[0];
rz(0.01418737340488748) q[1];
rz(0.1378972366643003) q[2];
rz(0.11977937689918428) q[3];
rx(-0.10461494042334178) q[0];
rx(-0.17867953546762322) q[1];
rx(-0.27133473111735606) q[2];
rx(-0.14149149444041073) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06318570233516839) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10603064573918644) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.057190932239058696) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2002296865585477) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.041160326872500416) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06985839502162448) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.045346492076025285) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.12454364712910966) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03701375294968304) q[3];
cx q[2],q[3];
rz(0.10724972108825305) q[0];
rz(-0.023721148882141656) q[1];
rz(0.12963253915032263) q[2];
rz(0.10054526418132019) q[3];
rx(-0.15055227782633201) q[0];
rx(-0.20447643168627788) q[1];
rx(-0.3140160283345126) q[2];
rx(-0.1636440814280539) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.11345057703012544) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.011242723798591429) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04450577144402972) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.14957024730395513) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.05746306144225823) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.03985991810422693) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.03950842151005266) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.01574525835898841) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.018596324360477007) q[3];
cx q[2],q[3];
rz(0.11567740195385633) q[0];
rz(0.014352739670824226) q[1];
rz(0.19896295004689948) q[2];
rz(0.10940801849369339) q[3];
rx(-0.08394907874942416) q[0];
rx(-0.14703882714159214) q[1];
rx(-0.17947191515602007) q[2];
rx(-0.27076785593316705) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0745493697894154) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.025678320115772552) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.05035163068531987) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1414571253019883) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.057152829191744164) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09812311174390488) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0897444670632756) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.043078029101832484) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.13603200431195206) q[3];
cx q[2],q[3];
rz(0.03652808950942835) q[0];
rz(-0.02839419359517257) q[1];
rz(0.16458908247378448) q[2];
rz(0.1734867454436884) q[3];
rx(-0.09929166305353145) q[0];
rx(-0.17936809210092855) q[1];
rx(-0.1974866899413139) q[2];
rx(-0.25895559456439243) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.00039616737140956375) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.10975429163249363) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03210171291413931) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.13424895050560512) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11400916601326584) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15755805421735686) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.06807647155977545) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.04242461366357543) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2649496500376995) q[3];
cx q[2],q[3];
rz(0.030631534620375572) q[0];
rz(-0.07616851873905008) q[1];
rz(0.1439927676104686) q[2];
rz(0.11192460056697096) q[3];
rx(-0.06080627065527401) q[0];
rx(-0.12870114603269162) q[1];
rx(-0.1372022349172317) q[2];
rx(-0.19363740152385966) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0562204616865015) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.11748569202723556) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.13720951569905598) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.20221553938230283) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.13243582272110624) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10370059964279327) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.04063985341692015) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.04343906252263845) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2591087419241049) q[3];
cx q[2],q[3];
rz(0.025125258649590993) q[0];
rz(-0.14498086030262985) q[1];
rz(0.13190360317557395) q[2];
rz(0.08617608120873617) q[3];
rx(0.005547139236071069) q[0];
rx(-0.16573102695424824) q[1];
rx(-0.17118280793672147) q[2];
rx(-0.13477430582489025) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.11154554955528403) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.10263614520057669) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.12832235025788138) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.20259256569183348) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.14525257093917954) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.05777385996341925) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08499089450357283) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.004418590883034091) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.23107596362683025) q[3];
cx q[2],q[3];
rz(0.04389217024128722) q[0];
rz(-0.176144225717614) q[1];
rz(0.07147283094016448) q[2];
rz(0.09278584733831016) q[3];
rx(-0.005844130590039511) q[0];
rx(-0.2930781694413731) q[1];
rx(-0.23833801963629656) q[2];
rx(-0.19665967677499427) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.011213804940215955) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.045529975857246774) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.05800458489572897) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2654990405651429) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.01919251106499597) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.09150859570430446) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.022416563519086188) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.12159559653935939) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2193269434229092) q[3];
cx q[2],q[3];
rz(-0.004763636016460266) q[0];
rz(-0.24510928182109662) q[1];
rz(-0.004182069978624383) q[2];
rz(0.04759124713523326) q[3];
rx(-0.027021898666630392) q[0];
rx(-0.25253151251922606) q[1];
rx(-0.24359024534659954) q[2];
rx(-0.10557813175091393) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.015301705759817085) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.02826892680030972) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07348224824103702) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.35464183935761806) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.06096619051896601) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.21687753619103278) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06067561729116009) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.11794288046557597) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.15887855496110245) q[3];
cx q[2],q[3];
rz(0.07549820305784258) q[0];
rz(-0.20923938271484574) q[1];
rz(-0.06028129707859421) q[2];
rz(0.046072868397646106) q[3];
rx(-0.024763540027600037) q[0];
rx(-0.2451085404296132) q[1];
rx(-0.29288684391555897) q[2];
rx(-0.0508546294574774) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1100059620395622) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13955183792873022) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.03280464265443181) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.43306050262922485) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.11180850695245631) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.21961419226082782) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.09707940677136731) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.04884877244369351) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09897929603217738) q[3];
cx q[2],q[3];
rz(0.0089800911069084) q[0];
rz(-0.1663147277867017) q[1];
rz(-0.04368627445838072) q[2];
rz(0.04170420032359933) q[3];
rx(-0.01994335052293821) q[0];
rx(-0.25881610436948616) q[1];
rx(-0.315048470283473) q[2];
rx(-0.01621517098287395) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2307638525025123) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.008781763535834286) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0031389073628916115) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.6078475628241009) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.11399445452605222) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.13392656124046004) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.15858705765709902) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.18942847690445166) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.061977262076101075) q[3];
cx q[2],q[3];
rz(-0.14996642150447015) q[0];
rz(-0.08339214386784524) q[1];
rz(0.10429309204367529) q[2];
rz(0.057782341972932555) q[3];
rx(-0.08297035887556427) q[0];
rx(-0.3167139129965717) q[1];
rx(-0.2449604059124728) q[2];
rx(-0.050689196587242975) q[3];