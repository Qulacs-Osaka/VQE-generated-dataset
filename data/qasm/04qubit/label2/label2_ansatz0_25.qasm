OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.004863252932783653) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09892013478265342) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.008632995583225793) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.34400490892882274) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.4626926562491291) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.03989343519503692) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.22048802206024148) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.3053667765330157) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.040091515126774266) q[3];
cx q[2],q[3];
rx(-0.14545104323277921) q[0];
rz(0.02696504912702655) q[0];
rx(-0.21762833386123254) q[1];
rz(-0.03675617757601406) q[1];
rx(0.32562385812811295) q[2];
rz(-0.12051600417726971) q[2];
rx(-0.2563099375533446) q[3];
rz(-0.14157596747964538) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.10190528302162355) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.14545293863715147) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.024765899857905064) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.1446173587521723) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.3899462326228133) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11111380734096543) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.11480058532121101) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.044600566644837004) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.14349844762684072) q[3];
cx q[2],q[3];
rx(-0.14371232335048229) q[0];
rz(0.10679400897064895) q[0];
rx(-0.2551236655793541) q[1];
rz(0.01104671715056836) q[1];
rx(0.26886945124833406) q[2];
rz(-0.12674196597792547) q[2];
rx(-0.22011275744994907) q[3];
rz(-0.08341235322454166) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.17769208874623824) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13199195871786917) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.018539771149852665) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.07418949877761429) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.17413081561321486) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07283071997605295) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.08763500117393921) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0019398627493544653) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.19189805574662114) q[3];
cx q[2],q[3];
rx(-0.10915327935293398) q[0];
rz(0.1339479433936716) q[0];
rx(-0.3328894069374169) q[1];
rz(-0.0473410411476379) q[1];
rx(0.3379316671292121) q[2];
rz(-0.13558854332790032) q[2];
rx(-0.24341616667198573) q[3];
rz(-0.15243202328992822) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.18384215655429142) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.16625278181831504) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.09391584245438588) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1494862243712004) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07051930539904126) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.002411519176643326) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05355925194110165) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.03295178164255383) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3119718242266184) q[3];
cx q[2],q[3];
rx(-0.13511095014166752) q[0];
rz(0.14380540197157374) q[0];
rx(-0.44318516356756005) q[1];
rz(-0.002591851776294261) q[1];
rx(0.2688685720318149) q[2];
rz(-0.18962234019980445) q[2];
rx(-0.09992357864545347) q[3];
rz(-0.14234413896158324) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.25583990756536124) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.23341174316936292) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.01703059447380323) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3972109480597966) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2015203187150116) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.12017706054528138) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.04906039873252408) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.14389818866148651) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.17360519191775167) q[3];
cx q[2],q[3];
rx(-0.07024436234534132) q[0];
rz(0.11485512720228434) q[0];
rx(-0.4371359822142072) q[1];
rz(0.06918755713304764) q[1];
rx(0.32948641107796994) q[2];
rz(-0.09073745083348131) q[2];
rx(-0.06810990623729418) q[3];
rz(-0.10096237617742783) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.23314609247849655) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.21853359297710845) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0466605285209091) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.4787116954540536) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2664149312319339) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2116842378196518) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.10549189821812582) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.09502362736033448) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.074820985746866) q[3];
cx q[2],q[3];
rx(-0.07452675713964368) q[0];
rz(0.10825973849531033) q[0];
rx(-0.48192699739636335) q[1];
rz(0.13620697177893798) q[1];
rx(0.270869168007312) q[2];
rz(-0.03704546988686903) q[2];
rx(0.039817759159564536) q[3];
rz(-0.058837819412171276) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.32770920486045607) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.14609581303977873) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.09405340462021013) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.45600453237232846) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.24118222010804008) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.17147590180968297) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.16098215544159103) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.16108213613813668) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04495586285688018) q[3];
cx q[2],q[3];
rx(-0.039859063831159086) q[0];
rz(0.026534923842322398) q[0];
rx(-0.5299459284099789) q[1];
rz(0.13311579064231766) q[1];
rx(0.33155441661619617) q[2];
rz(-0.04191324947230635) q[2];
rx(0.03946810525419477) q[3];
rz(-0.08113764644119563) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.3037780909830881) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06835304189061094) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.06278494329773339) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.46687925964162014) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.10001838920357099) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.005355660098312052) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.12443794335352389) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1703895489725225) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.14592816589684937) q[3];
cx q[2],q[3];
rx(-0.07326784652409876) q[0];
rz(0.038383417370659846) q[0];
rx(-0.44756919502619247) q[1];
rz(0.10927090612538254) q[1];
rx(0.28967798223889923) q[2];
rz(-0.04335441602084122) q[2];
rx(0.04431826895033133) q[3];
rz(-0.054303807307737796) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.34847365867947516) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.1115810137197693) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.13205341571496632) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3711449790714211) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.03876977707866894) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0327186902975979) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.110963257570561) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.12595253186598193) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.10787255555381317) q[3];
cx q[2],q[3];
rx(-0.09211880006637418) q[0];
rz(0.025439740480704062) q[0];
rx(-0.4559561910025162) q[1];
rz(0.06898160481279507) q[1];
rx(0.21489213420088712) q[2];
rz(0.02974386571885072) q[2];
rx(0.04233484062926971) q[3];
rz(-0.15685771371079585) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2641423928714111) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.12964527107901475) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.13952519042405498) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.23394563891734851) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.037395338493749264) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0002690998883314091) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08533649550977401) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.025189358893636885) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.15279054542727397) q[3];
cx q[2],q[3];
rx(-0.19951011867701296) q[0];
rz(-0.002204966094933965) q[0];
rx(-0.3766213612705848) q[1];
rz(0.06914861191920349) q[1];
rx(0.14422233203528018) q[2];
rz(0.04301942427040338) q[2];
rx(0.1336616224264872) q[3];
rz(-0.1661142056021725) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.31215274831642703) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.07116634433874094) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.25530031605343856) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.12849986877063227) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.053352546362408916) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.021104366578074893) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.026283904087955338) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.09241977360758787) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.18082474268564727) q[3];
cx q[2],q[3];
rx(-0.1324508701428875) q[0];
rz(-0.10481260187697965) q[0];
rx(-0.27662403159732896) q[1];
rz(0.03739086774415409) q[1];
rx(0.06149243140219217) q[2];
rz(0.07939191907973439) q[2];
rx(0.12479153930586087) q[3];
rz(-0.10590779311323058) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.18395916186956043) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.06251295221846258) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.3115846860388514) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.05797909181755774) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.12144390895408312) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.010724370707367866) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.047978680203937786) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.201854181458918) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.16399470173377872) q[3];
cx q[2],q[3];
rx(-0.18641654877955957) q[0];
rz(-0.06733734850135037) q[0];
rx(-0.18150471210358168) q[1];
rz(0.02060966711265486) q[1];
rx(-0.023495701950251843) q[2];
rz(0.04493125368594509) q[2];
rx(0.13019585288554722) q[3];
rz(-0.1544459061988416) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2244482223505864) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0044489130684380075) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.29015600622594834) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.023385869023369296) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2058267659256211) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07312488663445071) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.17633014451473641) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.34579054665305287) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.13720169060563886) q[3];
cx q[2],q[3];
rx(-0.2159271530309924) q[0];
rz(-0.07900007489097854) q[0];
rx(-0.015964094798436582) q[1];
rz(-0.015150943693880745) q[1];
rx(-0.04021938973718677) q[2];
rz(0.038782755448670186) q[2];
rx(0.23719469330809556) q[3];
rz(-0.18982951151474764) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2086273698905169) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.03652784684662667) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.23222099450143668) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.10449765497931371) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.35886891068700194) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0975222954966739) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.25213022911915733) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4764975945772848) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07961071015134527) q[3];
cx q[2],q[3];
rx(-0.16080507415293693) q[0];
rz(-0.08114104798543899) q[0];
rx(0.10347479698634093) q[1];
rz(-0.06545662870412239) q[1];
rx(-0.06004508017252826) q[2];
rz(0.11423260364356243) q[2];
rx(0.2971194500128765) q[3];
rz(-0.22439288644925856) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.12315392991277657) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06333344008093172) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.15780783858353095) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.24986810407654636) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.4183489068970546) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.08683838029045952) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.3178212713605474) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.5380431733888826) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.11187496444884874) q[3];
cx q[2],q[3];
rx(-0.25971679960095745) q[0];
rz(-0.1332638348783509) q[0];
rx(0.14782842014842035) q[1];
rz(-0.05687339210440467) q[1];
rx(-0.04643234501456309) q[2];
rz(0.19149459649788977) q[2];
rx(0.2532111606208521) q[3];
rz(-0.22275875897067055) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.10591123302058919) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.025282714367709222) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.09060916442250998) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.30685541668420596) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.37762400456376444) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2462097760255463) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.33194355697652) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4676529870803578) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.14886839877353705) q[3];
cx q[2],q[3];
rx(-0.2529649595931806) q[0];
rz(-0.10932126006680054) q[0];
rx(0.18771280732399326) q[1];
rz(-0.08693512312840872) q[1];
rx(-0.03900296460933435) q[2];
rz(0.20765716490077624) q[2];
rx(0.27528511774813785) q[3];
rz(-0.20721158640115445) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.15207775571025958) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.012338671830687984) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.035672536569668255) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.3023454767080454) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.375727333813962) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.32123622058721146) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.24722610023344274) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4603408053199341) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.11064171747705855) q[3];
cx q[2],q[3];
rx(-0.21470949192255004) q[0];
rz(-0.07946650500487086) q[0];
rx(0.19589649191946232) q[1];
rz(-0.08916989450089895) q[1];
rx(-0.20437157894854674) q[2];
rz(0.1773648519850067) q[2];
rx(0.30243823611386667) q[3];
rz(-0.26662197781219454) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.20936860782633232) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06761006973714262) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03708685458955168) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.3200063043767744) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.21283729543990285) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.5044433056134339) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.20402414863043322) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.36580072614228837) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06078745887464037) q[3];
cx q[2],q[3];
rx(-0.2782232096479006) q[0];
rz(-0.07108319286979187) q[0];
rx(0.2581527504465513) q[1];
rz(-0.08316912722889624) q[1];
rx(-0.3102998205698968) q[2];
rz(0.2875465823531984) q[2];
rx(0.2512435925063681) q[3];
rz(-0.28926524069184445) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2947657138093988) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.02995414850290091) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06885827188055794) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.2715644573393537) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.022840459546636813) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.5645828738243206) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.08117301886150506) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.3097763220328022) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.11403898218510887) q[3];
cx q[2],q[3];
rx(-0.27932369560792647) q[0];
rz(-0.04470466915210955) q[0];
rx(0.27883396535621885) q[1];
rz(-0.054344667268734846) q[1];
rx(-0.24385370721920677) q[2];
rz(0.20588225881591257) q[2];
rx(0.290169344076514) q[3];
rz(-0.26247135613967204) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.27288345927056323) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.06760156018592922) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.05578500274259862) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.14709112344325734) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.0844198227773639) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.6178653416302937) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.08952173317283525) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.37739758090744946) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1402907835932729) q[3];
cx q[2],q[3];
rx(-0.24529948907164562) q[0];
rz(-0.08682957165711473) q[0];
rx(0.2928146974505985) q[1];
rz(-0.0595551161771341) q[1];
rx(-0.22989872335307332) q[2];
rz(0.22993136035621187) q[2];
rx(0.2488170656429381) q[3];
rz(-0.13337305492383641) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.18546326912292455) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.05140974522656805) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09922677944538127) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.027273147566177176) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.162996550385217) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.6602719387495007) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0645033337471202) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.3513312046528654) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.15885162625849356) q[3];
cx q[2],q[3];
rx(-0.24223650314288492) q[0];
rz(-0.046091050894154544) q[0];
rx(0.21778670186870994) q[1];
rz(0.011731089882309741) q[1];
rx(-0.17255508626113994) q[2];
rz(0.19535667853470293) q[2];
rx(0.20126693539015475) q[3];
rz(-0.06275210921309929) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.09041874579368551) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.05033349676933956) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.074706654295799) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.06339132597018428) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.23666233305184586) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.6834257140395138) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0947230059445545) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.378526479130987) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.059428104781585514) q[3];
cx q[2],q[3];
rx(-0.2314789399427403) q[0];
rz(-0.0632698713774761) q[0];
rx(0.20371761229586755) q[1];
rz(0.07195370682825485) q[1];
rx(-0.1938770683265951) q[2];
rz(0.100241077372453) q[2];
rx(0.1281918880522382) q[3];
rz(-0.022349967629574946) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03823572970030592) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.04298935595941185) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0920862340190024) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.083082407883783) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.32289762641842507) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.7076704656617817) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.035233057141870645) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.3626488537751382) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05821069640946395) q[3];
cx q[2],q[3];
rx(-0.22004365184858796) q[0];
rz(-0.12115984636413915) q[0];
rx(0.19466433984789527) q[1];
rz(0.06482531594505687) q[1];
rx(-0.24106468083058424) q[2];
rz(-0.079465082650701) q[2];
rx(0.15744747960562572) q[3];
rz(0.03423900072235239) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1045667202011943) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.03960717469777101) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.10594267591914162) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.17683680164271912) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3943740211110423) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.6344095932545829) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09579498198163236) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.25693552264191566) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05546717093857484) q[3];
cx q[2],q[3];
rx(-0.1816888971209349) q[0];
rz(-0.210247106154136) q[0];
rx(0.09982320429423663) q[1];
rz(0.200231999897755) q[1];
rx(-0.2589778200754134) q[2];
rz(-0.22164982574800354) q[2];
rx(0.10169237551410397) q[3];
rz(0.1669669180466529) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.01346855177394125) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.1020340586962769) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.17999328065362966) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3141807267618928) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.38323185306149904) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.4613233096473922) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0020562693650061684) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.09252635639336647) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0015335511470422637) q[3];
cx q[2],q[3];
rx(-0.05122839678103781) q[0];
rz(-0.21410899367193595) q[0];
rx(-0.06158079026428324) q[1];
rz(0.21167375499687033) q[1];
rx(-0.24037553909648054) q[2];
rz(-0.33659712235092004) q[2];
rx(0.11734646321375149) q[3];
rz(0.218964648671219) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.045473949959006284) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.19754573349331458) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.11586186983944582) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3514990054773415) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2713758342201914) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.3014222438077504) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.010468434804810746) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.01982502958347259) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07369667307030627) q[3];
cx q[2],q[3];
rx(-0.023880671053744324) q[0];
rz(-0.3256793933820889) q[0];
rx(-0.14173125263946842) q[1];
rz(0.20339506623234724) q[1];
rx(-0.21138854719207148) q[2];
rz(-0.4976934976276724) q[2];
rx(0.04555287184181575) q[3];
rz(0.37846044166415915) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0967510708990631) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.10556552591367972) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.14693310566153403) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.5535134140970458) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.10822101658250376) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.008340054152514187) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.08755184314798714) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.12934094932519696) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07837119186640985) q[3];
cx q[2],q[3];
rx(0.06746856501422792) q[0];
rz(-0.3875325525966419) q[0];
rx(-0.19758228678013037) q[1];
rz(0.35261791645441404) q[1];
rx(-0.16858962499628335) q[2];
rz(-0.47814930100407227) q[2];
rx(0.11777737192905173) q[3];
rz(0.35374708699220697) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05057370657098288) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.08777576673735508) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.12334627968668764) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.6251605951569008) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.10302869465371593) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.13407730807334337) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0005075808556257895) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.08883305747866181) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.14230092057535035) q[3];
cx q[2],q[3];
rx(0.10803957884753063) q[0];
rz(-0.4489059748657443) q[0];
rx(-0.27495163693962216) q[1];
rz(0.27267196644852154) q[1];
rx(0.03287904678405296) q[2];
rz(-0.5486408399133538) q[2];
rx(-0.02762379106526892) q[3];
rz(0.43384883548280406) q[3];