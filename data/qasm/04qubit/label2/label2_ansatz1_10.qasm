OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.22596081082910854) q[0];
rz(2.3042389877508183) q[0];
ry(0.26656380220717774) q[1];
rz(0.5859216288464684) q[1];
ry(-2.63542670409871) q[2];
rz(0.3649034842010801) q[2];
ry(0.3614588249579853) q[3];
rz(2.277266869877103) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.7589768344816404) q[0];
rz(-0.37140444307410553) q[0];
ry(-1.290971431644718) q[1];
rz(-1.5397988547970272) q[1];
ry(-2.0568156715168495) q[2];
rz(-2.2790872183813016) q[2];
ry(-2.0697862432640024) q[3];
rz(-0.5220247062227248) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.674495872958123) q[0];
rz(-2.572243909731515) q[0];
ry(1.7359513249909868) q[1];
rz(3.082479926698926) q[1];
ry(3.110267670512691) q[2];
rz(1.1773563307915889) q[2];
ry(1.4607822964699817) q[3];
rz(-1.4793824509788338) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.007600756810474) q[0];
rz(1.0351462000607095) q[0];
ry(1.939122041853615) q[1];
rz(0.8933222110697442) q[1];
ry(-0.2865046408533738) q[2];
rz(0.461916719697971) q[2];
ry(2.1271835350199355) q[3];
rz(0.7065939558412321) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.07074499073835307) q[0];
rz(-1.1345099464891133) q[0];
ry(-2.4559625695094534) q[1];
rz(2.046363395560391) q[1];
ry(0.8134814605673948) q[2];
rz(1.4515095284487665) q[2];
ry(2.7091940116417494) q[3];
rz(2.1544902968911) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.3416468135308035) q[0];
rz(2.3061504471204017) q[0];
ry(2.6226600296605995) q[1];
rz(-0.1360658825994213) q[1];
ry(0.12269893690808406) q[2];
rz(-1.1475239035416163) q[2];
ry(-1.9887171372082486) q[3];
rz(-1.096744424901412) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.6339756081191819) q[0];
rz(-0.10379302862716866) q[0];
ry(2.794199561306553) q[1];
rz(-1.2726280059105848) q[1];
ry(1.4416309310131075) q[2];
rz(0.4158375477349488) q[2];
ry(-1.9720606009239958) q[3];
rz(-0.05379061574018472) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.4299660676410815) q[0];
rz(0.6401086824416016) q[0];
ry(1.4237141475170223) q[1];
rz(-0.8202761620441645) q[1];
ry(0.5805416564399399) q[2];
rz(-2.202435477562431) q[2];
ry(-0.4036825747345906) q[3];
rz(0.5787347187448493) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.837426829909883) q[0];
rz(2.48594933126703) q[0];
ry(-2.6826994586381296) q[1];
rz(-0.27422500607089884) q[1];
ry(0.9300988977652187) q[2];
rz(-1.2459701927925533) q[2];
ry(1.959117200439028) q[3];
rz(-0.5569962082370782) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.8552636773533173) q[0];
rz(-2.6103060580636206) q[0];
ry(-1.9953698918474245) q[1];
rz(-2.9489512996487752) q[1];
ry(-2.548924747888844) q[2];
rz(0.8777949951001868) q[2];
ry(0.6316909571833229) q[3];
rz(-0.2675629359055174) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8712747528237719) q[0];
rz(0.15680615198094155) q[0];
ry(3.012834814707924) q[1];
rz(-2.837030728446211) q[1];
ry(-1.6230645377528168) q[2];
rz(-2.5289140471529237) q[2];
ry(0.3720113894001629) q[3];
rz(0.8991571853615995) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.8182792661893714) q[0];
rz(1.6316251142623042) q[0];
ry(-1.0314029124474218) q[1];
rz(2.0092476025171297) q[1];
ry(3.0492216184384735) q[2];
rz(-1.2189067625517467) q[2];
ry(-1.0063557026041359) q[3];
rz(-2.5842783627698447) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.39375022777580454) q[0];
rz(1.9105406327778436) q[0];
ry(-0.49167113088652903) q[1];
rz(0.6855181524914018) q[1];
ry(3.1354156567893057) q[2];
rz(-0.22686607728744443) q[2];
ry(3.052089228978771) q[3];
rz(-2.465047167270953) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.6693180023814662) q[0];
rz(2.510614169319072) q[0];
ry(2.4462415308102026) q[1];
rz(0.03938122842348601) q[1];
ry(2.8692763649945374) q[2];
rz(1.331786852222602) q[2];
ry(0.9040049485215575) q[3];
rz(-1.9373718138435176) q[3];