OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5099995638796495) q[0];
rz(0.02867900771005967) q[0];
ry(2.6841765894240495) q[1];
rz(1.0122572519986024) q[1];
ry(3.141566303310345) q[2];
rz(-0.9761439233948352) q[2];
ry(-3.126001237680595) q[3];
rz(0.7165926559524342) q[3];
ry(-0.0615745851849315) q[4];
rz(-1.6401072141804798) q[4];
ry(0.23899826840506008) q[5];
rz(0.7932837905443048) q[5];
ry(-0.8397926650960033) q[6];
rz(1.4876159321816984) q[6];
ry(-1.0090117580767055) q[7];
rz(2.63738897640601) q[7];
ry(-2.107486692229035) q[8];
rz(2.91643036327346) q[8];
ry(2.3745576117232186) q[9];
rz(1.3632353127339805) q[9];
ry(-1.570524870210554) q[10];
rz(0.00018028590956850653) q[10];
ry(-1.5708587109746766) q[11];
rz(-3.140959911671996) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5477473032720135) q[0];
rz(0.10795851133738832) q[0];
ry(-0.00107954614688488) q[1];
rz(-0.24610130367334973) q[1];
ry(-3.1415851193593882) q[2];
rz(1.8742379076400615) q[2];
ry(2.968510517551186) q[3];
rz(-0.9961543102252994) q[3];
ry(2.613297395420509) q[4];
rz(1.5236966488541126) q[4];
ry(2.160568521270083) q[5];
rz(1.3043902901241444) q[5];
ry(-2.917353882661857) q[6];
rz(1.5187295860668035) q[6];
ry(-3.052258875131318) q[7];
rz(-2.2819899546565203) q[7];
ry(0.3971982688820761) q[8];
rz(1.0103723044980841) q[8];
ry(0.014288444941943368) q[9];
rz(1.037964187886598) q[9];
ry(-2.4373535353958484) q[10];
rz(1.5708573875376584) q[10];
ry(-2.530812969256036) q[11];
rz(1.572011898563507) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.019742872670581946) q[0];
rz(-1.7962155302160432) q[0];
ry(3.140085011927257) q[1];
rz(-2.3163008545716597) q[1];
ry(2.536939601715505e-06) q[2];
rz(-2.0475938842377213) q[2];
ry(0.4867256196331313) q[3];
rz(-1.6072341986286023) q[3];
ry(-1.372624245691558) q[4];
rz(0.7285092720401702) q[4];
ry(-2.8999696119093183) q[5];
rz(-2.8912904510900765) q[5];
ry(-0.03482735445631757) q[6];
rz(-1.6445706320761424) q[6];
ry(-3.0634571358041116) q[7];
rz(1.4537371064856197) q[7];
ry(-0.14939710860311095) q[8];
rz(2.033924381472456) q[8];
ry(-3.1402879164040325) q[9];
rz(1.9319264253202677) q[9];
ry(-0.5735121643290101) q[10];
rz(1.5688561675284838) q[10];
ry(2.803100625951094) q[11];
rz(-1.0241905109815235) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.9589564996584556) q[0];
rz(-1.679033111616837) q[0];
ry(1.5774565388096988) q[1];
rz(-3.1401295128990507) q[1];
ry(3.141576598061333) q[2];
rz(-1.9985277145194527) q[2];
ry(-1.8375492440541166) q[3];
rz(-1.993731865793615) q[3];
ry(1.3183841394512834) q[4];
rz(2.2794193164014156) q[4];
ry(2.9444476436968294) q[5];
rz(0.9459156745826842) q[5];
ry(0.007483766380051013) q[6];
rz(-1.4836251023371023) q[6];
ry(-3.0684075166045597) q[7];
rz(0.8042484501206271) q[7];
ry(3.036051853087463) q[8];
rz(-1.5247356656810904) q[8];
ry(0.007373311517683057) q[9];
rz(-3.058647759099838) q[9];
ry(-0.1338316214389872) q[10];
rz(-1.5686421801043562) q[10];
ry(0.00023888647095930514) q[11];
rz(1.0356229558308119) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5712773590605953) q[0];
rz(-1.6032957535939818) q[0];
ry(1.5759393445547645) q[1];
rz(0.10472998163244363) q[1];
ry(-1.1073881807099895e-05) q[2];
rz(-2.301150504499386) q[2];
ry(-0.0014022028101553581) q[3];
rz(-1.7307858708011823) q[3];
ry(0.005519537120473011) q[4];
rz(2.348865016651939) q[4];
ry(-0.006944389670094844) q[5];
rz(-0.1093102771799561) q[5];
ry(3.140697664798238) q[6];
rz(2.998318967398571) q[6];
ry(-3.141570248110654) q[7];
rz(0.21591843442244113) q[7];
ry(3.137395184344977) q[8];
rz(-1.6563958168755297) q[8];
ry(0.0006875565542667772) q[9];
rz(2.479936043570743) q[9];
ry(-1.6377885379281594) q[10];
rz(-1.6259286981178862) q[10];
ry(3.0470245158686913) q[11];
rz(1.5814853113604856) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.3974810521395535) q[0];
rz(-3.139624633570876) q[0];
ry(-1.2188600884816154) q[1];
rz(1.016657413412271) q[1];
ry(3.1415260180440416) q[2];
rz(0.8221359168748857) q[2];
ry(-3.104207881951535) q[3];
rz(-0.5364371464641042) q[3];
ry(-3.119151915523606) q[4];
rz(2.8144004479812077) q[4];
ry(-0.07703362634353539) q[5];
rz(-0.9342998367689583) q[5];
ry(-0.030498166927720765) q[6];
rz(3.0243187764908432) q[6];
ry(0.010582465528416093) q[7];
rz(-3.1309335788671557) q[7];
ry(0.0012611533520612284) q[8];
rz(-1.5552166930868248) q[8];
ry(-3.1410302431930113) q[9];
rz(1.7724845326735998) q[9];
ry(0.002155351530600519) q[10];
rz(-1.5153056278912072) q[10];
ry(1.5441873547112361) q[11];
rz(1.569884122545209) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.49107938231148207) q[0];
rz(2.0753734814350278) q[0];
ry(3.0995797727735606) q[1];
rz(1.3964273800720708) q[1];
ry(1.570790767423537) q[2];
rz(-4.7455011289443405e-06) q[2];
ry(-1.642570982063841) q[3];
rz(-0.5494061980508507) q[3];
ry(-0.4484434823969816) q[4];
rz(1.5160824617428315) q[4];
ry(3.1122130676660826) q[5];
rz(1.933673375450912) q[5];
ry(0.030902389279519937) q[6];
rz(2.898576085194972) q[6];
ry(-0.022604575995106124) q[7];
rz(0.30524740789536686) q[7];
ry(-0.00927897336306795) q[8];
rz(-1.0725372200294325) q[8];
ry(0.0003196992933940379) q[9];
rz(1.7591248856990174) q[9];
ry(-0.3510096637436995) q[10];
rz(-1.5727115389452437) q[10];
ry(-0.32659447037885164) q[11];
rz(-1.568706042292301) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-4.7238576217432104e-05) q[0];
rz(2.349961405082923) q[0];
ry(3.1415851260933585) q[1];
rz(2.7472740339887203) q[1];
ry(-1.5708363979143123) q[2];
rz(0.18920551301835523) q[2];
ry(6.154777731025423e-05) q[3];
rz(-0.31346726345865744) q[3];
ry(-3.141480533164377) q[4];
rz(1.576944241338599) q[4];
ry(-3.141580651363491) q[5];
rz(-1.791271027559353) q[5];
ry(-5.5803368201018334e-05) q[6];
rz(-2.5179134691693066) q[6];
ry(-0.00019070776081608898) q[7];
rz(-3.008050509678091) q[7];
ry(1.570337316387112) q[8];
rz(-0.00012842469156915112) q[8];
ry(3.1415211391028706) q[9];
rz(1.4611533723313395) q[9];
ry(0.30927675153443607) q[10];
rz(0.7713145790029383) q[10];
ry(0.3095781693391819) q[11];
rz(-0.7686780938193216) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.0026188035609999716) q[0];
rz(1.3150926649014958) q[0];
ry(-0.017856835025197686) q[1];
rz(3.0962863964808105) q[1];
ry(3.1345620059768624) q[2];
rz(-2.240668008720073) q[2];
ry(-3.1200450460727094) q[3];
rz(-1.0273334193725692) q[3];
ry(0.08795195007800059) q[4];
rz(-0.10930158067101182) q[4];
ry(2.7724254583061194) q[5];
rz(-1.2698389864826936) q[5];
ry(2.450003891263237) q[6];
rz(-2.851385528764392) q[6];
ry(2.225684381125549) q[7];
rz(2.9225001442492147) q[7];
ry(1.5709782875466525) q[8];
rz(-2.7279581667943) q[8];
ry(-0.3926370282983749) q[9];
rz(3.0817217290319627) q[9];
ry(-3.139683224114008) q[10];
rz(-0.48616454527523423) q[10];
ry(3.139720332103598) q[11];
rz(2.3228763371268877) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.139248875899124) q[0];
rz(-0.8875861975099046) q[0];
ry(3.137778085086774) q[1];
rz(1.777210553656639) q[1];
ry(-3.1355130026344944) q[2];
rz(1.182786277339777) q[2];
ry(3.0869248834325704) q[3];
rz(0.22002899063608883) q[3];
ry(-0.15618816129757018) q[4];
rz(2.312049174220281) q[4];
ry(2.923393988457052) q[5];
rz(-2.845679132142275) q[5];
ry(2.0564060477276884) q[6];
rz(0.24856361867172594) q[6];
ry(-1.4155377301388237) q[7];
rz(0.04437901963681991) q[7];
ry(0.0004321142563410231) q[8];
rz(1.6999825309018881) q[8];
ry(-0.5372297244284687) q[9];
rz(2.6631158196944607) q[9];
ry(-3.11990418289433) q[10];
rz(0.3251517693044713) q[10];
ry(1.8690549447228628) q[11];
rz(-0.5979279988139767) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.0025262916698259507) q[0];
rz(-0.06311056714886992) q[0];
ry(-3.140984305564728) q[1];
rz(-0.5751824943259908) q[1];
ry(-3.1358163157015055) q[2];
rz(0.35181459146656285) q[2];
ry(3.1256113982640765) q[3];
rz(-3.0616799413274856) q[3];
ry(3.131791217063612) q[4];
rz(0.09408970927045686) q[4];
ry(-3.1281760321236836) q[5];
rz(-3.059181228310756) q[5];
ry(2.9618846653848925) q[6];
rz(0.20762348724593682) q[6];
ry(0.7782232632970832) q[7];
rz(-0.7671177803530959) q[7];
ry(-1.5699067615562896) q[8];
rz(-3.14113147478422) q[8];
ry(3.0389684982833542) q[9];
rz(2.881935096998769) q[9];
ry(-3.0803067318971933) q[10];
rz(-1.9899611879689436) q[10];
ry(-3.1397658218482865) q[11];
rz(-1.7192848290307972) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.1415093515652672) q[0];
rz(0.011530726502094526) q[0];
ry(2.361640782488905e-06) q[1];
rz(-1.2982717080499926) q[1];
ry(6.679172336085069e-05) q[2];
rz(1.5975173996288887) q[2];
ry(4.498838904694377e-05) q[3];
rz(-1.60725356046296) q[3];
ry(3.42733105220816e-05) q[4];
rz(1.3159771289761124) q[4];
ry(9.458257673955472e-06) q[5];
rz(-2.869681645344082) q[5];
ry(-3.141501130607481) q[6];
rz(-2.245785696048885) q[6];
ry(0.0006827739980472814) q[7];
rz(-2.463162053408831) q[7];
ry(-1.5715739354956098) q[8];
rz(-2.060780417880519) q[8];
ry(-3.1413509746382777) q[9];
rz(1.9607468940805681) q[9];
ry(1.5501386148232796) q[10];
rz(1.5803516497912564) q[10];
ry(1.560960528369065) q[11];
rz(-1.5499552851609204) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1413935117824336) q[0];
rz(-1.5832930249694188) q[0];
ry(6.651730260731625e-05) q[1];
rz(2.4276428498767992) q[1];
ry(5.6522047825247064e-05) q[2];
rz(2.7727501125230245) q[2];
ry(1.035276575224033e-05) q[3];
rz(-0.16857770765982769) q[3];
ry(3.1415102651003926) q[4];
rz(-2.8742773498830028) q[4];
ry(-3.1415370998741836) q[5];
rz(1.2954723206639933) q[5];
ry(3.1414544345246465) q[6];
rz(-0.9329762763380441) q[6];
ry(-0.0006757837339232965) q[7];
rz(1.2952342193080586) q[7];
ry(-3.1414056801899197) q[8];
rz(2.3772633564048022) q[8];
ry(-3.1413821621208) q[9];
rz(-0.1803758289997832) q[9];
ry(-1.5708763738858735) q[10];
rz(-0.31078838102892803) q[10];
ry(1.5708608346321071) q[11];
rz(2.831900182469224) q[11];