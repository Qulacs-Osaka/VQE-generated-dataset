OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.15728253675621942) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.07992637701211354) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09584201947697625) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.14290715638145235) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.02527951455655369) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0529726197806149) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.15522839517833145) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.06614186451180758) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0021706812236172214) q[3];
cx q[2],q[3];
rz(-0.07192416888399414) q[0];
rz(-0.10492856237482563) q[1];
rz(-0.09464542515380817) q[2];
rz(-0.03135851298328544) q[3];
rx(-0.2687465640797199) q[0];
rx(0.1905187581943763) q[1];
rx(-0.44037107831597444) q[2];
rx(-0.2045048556764144) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0643350715832524) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.014255175851417114) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0444852843556289) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.06365341231378253) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.09157022961218708) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.010709397155931891) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08325686889691043) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.034507299799514626) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.00911068807704799) q[3];
cx q[2],q[3];
rz(-0.05066379324647542) q[0];
rz(-0.018469397561332316) q[1];
rz(-0.11295043878555758) q[2];
rz(-0.05072764599934349) q[3];
rx(-0.20933269658639497) q[0];
rx(0.20266477160706528) q[1];
rx(-0.3819642909616575) q[2];
rx(-0.1737732053902726) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06787531656140748) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.05790785317631549) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.028533207182352097) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.07071186422306505) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.15547594255778013) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.023112146393930416) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.10435325187810156) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1340814237989135) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0022937772050995366) q[3];
cx q[2],q[3];
rz(-0.05965699838916904) q[0];
rz(-0.026490981726299993) q[1];
rz(-0.12048959806728744) q[2];
rz(-0.042162496658685926) q[3];
rx(-0.17090800271850898) q[0];
rx(0.05039176832116548) q[1];
rx(-0.32478013792016025) q[2];
rx(-0.21686097415458572) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0390237604587906) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11770263854390295) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.07763928913302917) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.039792463926438607) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2268678074870873) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07560965876362301) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0050856850357053375) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.13364260828224628) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07053559352701201) q[3];
cx q[2],q[3];
rz(-0.03843311434607413) q[0];
rz(0.0749823694329208) q[1];
rz(-0.013851727967614807) q[2];
rz(-0.07003673704444474) q[3];
rx(-0.2057714579445028) q[0];
rx(-0.10105739799342754) q[1];
rx(-0.2772995964047271) q[2];
rx(-0.2484399287359044) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03753266288242434) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.08343949937592188) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.034512384957016654) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.056549491226915743) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.24584342847656762) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0044229409418469565) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06607992875535874) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.23556610658438992) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04853911194347922) q[3];
cx q[2],q[3];
rz(-0.09273173365679611) q[0];
rz(0.08080880467371476) q[1];
rz(-0.054277566697936194) q[2];
rz(0.018431553549944534) q[3];
rx(-0.21061720743749826) q[0];
rx(-0.10664055550190751) q[1];
rx(-0.19828444370447587) q[2];
rx(-0.2848367385382438) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.04021264310077915) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.17242366993819846) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.018824320191365814) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08974818306034124) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.12406839496098639) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.13029535065518463) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.11956503095036199) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2638222916515967) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07174850532735665) q[3];
cx q[2],q[3];
rz(-0.025733166593867894) q[0];
rz(0.1066531819579468) q[1];
rz(0.022909995527940902) q[2];
rz(0.03004974481770743) q[3];
rx(-0.1812288159899932) q[0];
rx(-0.2557762841458237) q[1];
rx(-0.16832245598601536) q[2];
rx(-0.31170252623941685) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.057773106910507216) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11375774089371848) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.047243832818192055) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.11248726799223045) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.0672055098505064) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.18328551303535043) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.08097501159441264) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2820949711862754) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1485635198124446) q[3];
cx q[2],q[3];
rz(-0.05629277230546672) q[0];
rz(0.1331730908977053) q[1];
rz(0.10142040190542785) q[2];
rz(0.03459907095880257) q[3];
rx(-0.12402978281323185) q[0];
rx(-0.2690488396108015) q[1];
rx(-0.08584807265780425) q[2];
rx(-0.30214044612742536) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0041305920141327225) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.12756875328243736) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.01615256146483571) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.09039994484066749) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.10021598309912357) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10734481073648527) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0720325398977567) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.27822049013242706) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.168442875745441) q[3];
cx q[2],q[3];
rz(0.007940089045120455) q[0];
rz(0.10529743130930443) q[1];
rz(0.0971993611888813) q[2];
rz(0.09748964608681837) q[3];
rx(-0.12003824098007151) q[0];
rx(-0.28622429928102433) q[1];
rx(-0.08633662035764811) q[2];
rx(-0.26046646188258465) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.010918022434739588) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.08388185342673117) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.05427549972512614) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08905605755988606) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.13759543847981992) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.16075838954499377) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.1673674037721204) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2964346883461071) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.09174286311932961) q[3];
cx q[2],q[3];
rz(0.008148774545568062) q[0];
rz(0.10938577389817555) q[1];
rz(0.14673670423753446) q[2];
rz(0.1806230476680707) q[3];
rx(-0.13316350833253146) q[0];
rx(-0.3300038282323605) q[1];
rx(-0.073602413389799) q[2];
rx(-0.28370374964029377) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.02237889866920477) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10926425394914105) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.05269477469468825) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.1157211458871225) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.1381308092985406) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.12453094948255138) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.08416553268785996) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.29034392626809574) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.053567225089887247) q[3];
cx q[2],q[3];
rz(0.018456674155956806) q[0];
rz(0.07234967973876193) q[1];
rz(0.1409124472957363) q[2];
rz(0.15438350815122845) q[3];
rx(-0.06924947636659973) q[0];
rx(-0.33256930841843496) q[1];
rx(-0.16762386596186518) q[2];
rx(-0.27518626262213214) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.052501655591743766) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.007415779091772582) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.02921464223754666) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.07916118278629217) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.062110848548196554) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06345843243175621) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.03987632324085934) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2506974475760281) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.040253798970173886) q[3];
cx q[2],q[3];
rz(0.051562454778587806) q[0];
rz(0.11836868343620527) q[1];
rz(0.14885736029968955) q[2];
rz(0.15793869187820075) q[3];
rx(-0.08181263015677237) q[0];
rx(-0.25238412861761716) q[1];
rx(-0.1541962193922804) q[2];
rx(-0.28378452596648945) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07037338022419443) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.013765703600556997) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.08657941722620603) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.07238725756629588) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.043368676174310755) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.03328611909081912) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0824133725474657) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.15626748196816168) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05705783541641948) q[3];
cx q[2],q[3];
rz(0.06377317830826972) q[0];
rz(0.21835717108291683) q[1];
rz(0.15939201642913192) q[2];
rz(0.18356830351538328) q[3];
rx(-0.08458764779177935) q[0];
rx(-0.2503277329370792) q[1];
rx(-0.12764144760430918) q[2];
rx(-0.3302138063070306) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.00620229138355352) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.036067244700980786) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03673765903860102) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.0739752365846524) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.12070574433411269) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.011108224598855973) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.028513457492962045) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1561862536886493) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.10263323333831989) q[3];
cx q[2],q[3];
rz(0.09885826688071297) q[0];
rz(0.15336474896109703) q[1];
rz(0.21836220060272835) q[2];
rz(0.18100023655933944) q[3];
rx(-0.06418763300824205) q[0];
rx(-0.18817297664909763) q[1];
rx(-0.1518007345925223) q[2];
rx(-0.2645475405042502) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.057126018060691675) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.02074895100112416) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.012009508821369886) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.10973102071796402) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.06995982807454819) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.1144876676795136) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.036874535368752376) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.09351167849024954) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.09260595658852495) q[3];
cx q[2],q[3];
rz(0.08210161522385374) q[0];
rz(0.22609226864463558) q[1];
rz(0.2401190156344467) q[2];
rz(0.11630333954613793) q[3];
rx(-0.09838577883443425) q[0];
rx(-0.2210846155581283) q[1];
rx(-0.16860347239060045) q[2];
rx(-0.28389734061982874) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.036010684766289894) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.037122155210956624) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.009856532533088202) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.11288414803527994) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.12668981888051212) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.053738364577043715) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07999473862794133) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.010555516333719083) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1545069992744531) q[3];
cx q[2],q[3];
rz(0.08150611404083988) q[0];
rz(0.16390587054309902) q[1];
rz(0.18607072632910238) q[2];
rz(0.07822253454112825) q[3];
rx(-0.04570729374080821) q[0];
rx(-0.18417529332401725) q[1];
rx(-0.1977030965425732) q[2];
rx(-0.29241714827424514) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.013685225169581934) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0016118636321922142) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04028042108628141) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.056061684018476346) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.12498015094190573) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.11487703650109263) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1449181590795232) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0026245875351610954) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.14811959412332248) q[3];
cx q[2],q[3];
rz(0.1426861136522208) q[0];
rz(0.25077989422351105) q[1];
rz(0.22531637745136365) q[2];
rz(0.09898509197824452) q[3];
rx(-0.04728845173929207) q[0];
rx(-0.29891570722574223) q[1];
rx(-0.2495613218733659) q[2];
rx(-0.2961385195851038) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0662910085684991) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.04715124191661474) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.001225287610088625) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.08646313089650745) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.059265279789962905) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.08279222296675588) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.14030161917732717) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.07572297929311553) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.058260282648512005) q[3];
cx q[2],q[3];
rz(0.1539311987780361) q[0];
rz(0.2497247337728881) q[1];
rz(0.15511894570153148) q[2];
rz(0.1332396652356091) q[3];
rx(-0.12236363512085141) q[0];
rx(-0.21757349440593257) q[1];
rx(-0.24089460013487343) q[2];
rx(-0.24816107665524986) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.008318648139005664) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07166931314812666) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-8.31512410846273e-05) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1451587041716173) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.02130646624381218) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15452718144923) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.13635274981033974) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.06059824851213097) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02653159086509062) q[3];
cx q[2],q[3];
rz(0.0752274575946252) q[0];
rz(0.19995925395313163) q[1];
rz(0.12991397196347648) q[2];
rz(0.050074333779332514) q[3];
rx(-0.0729355238205746) q[0];
rx(-0.1582827246087112) q[1];
rx(-0.331508906938054) q[2];
rx(-0.15433347109814774) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.05020113167309775) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.024557061641961104) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.02725161729449043) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.20968399243486263) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11316305972270457) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.19939300697327203) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.14650843204016392) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.14084212622789652) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.008761058957903262) q[3];
cx q[2],q[3];
rz(0.06373069389145307) q[0];
rz(0.11958022979735214) q[1];
rz(0.001107324147439827) q[2];
rz(0.037686200076157944) q[3];
rx(-0.06287646876842028) q[0];
rx(-0.1671972242110153) q[1];
rx(-0.28053051565330916) q[2];
rx(-0.1541695376303989) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.03955160278650894) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1075886622383914) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.03788093571602992) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2355979220222635) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.15436350156383194) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2686019070316442) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.21302860081838887) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.11643880095672321) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.01744293730960418) q[3];
cx q[2],q[3];
rz(-0.0030316400584739877) q[0];
rz(0.10647015410036326) q[1];
rz(-0.08135051730190893) q[2];
rz(-0.010565958284140439) q[3];
rx(-0.08807957405817056) q[0];
rx(-0.1562244614561193) q[1];
rx(-0.3295699820748742) q[2];
rx(-0.15952877138240554) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.018055051161075843) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.15486554407070893) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.02776930930259358) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.24679407680712398) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.16621652674528786) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.22656232387184677) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.23763668065700946) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.14349750124085434) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05224855841878879) q[3];
cx q[2],q[3];
rz(-0.0325995921603992) q[0];
rz(0.07507925349384402) q[1];
rz(-0.16338241825587224) q[2];
rz(-0.11539864441348051) q[3];
rx(-0.05971750317572359) q[0];
rx(-0.12610004831971883) q[1];
rx(-0.3557260239181592) q[2];
rx(-0.10199938561439896) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.04244024832367874) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.15267480567858147) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.048153915823639176) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2252390600464933) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.15840012731391442) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.20133203359695628) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2960655543273803) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.09562085776134746) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09225559364311038) q[3];
cx q[2],q[3];
rz(-0.08917528412805081) q[0];
rz(-0.03287551966260624) q[1];
rz(-0.23458025886693679) q[2];
rz(-0.21476656716563583) q[3];
rx(-0.051392245221996286) q[0];
rx(-0.14092900239606837) q[1];
rx(-0.3082776063914383) q[2];
rx(0.004616169382742218) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05442748339073109) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11787095012581397) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.05248135552457023) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.21822481833920873) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.019097694675928024) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1301956484204232) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.33688970889735825) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.09115222834574606) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09610557213132902) q[3];
cx q[2],q[3];
rz(-0.15421901475029243) q[0];
rz(-0.018936526105681747) q[1];
rz(-0.2583799664690322) q[2];
rz(-0.1889146782774187) q[3];
rx(-0.05580191853971295) q[0];
rx(-0.034254260466835405) q[1];
rx(-0.24447840111908353) q[2];
rx(-0.03437842142249754) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.023654303446241108) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07512933571366301) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.05561821641216308) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1409428123333812) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.019502667725572357) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.12588521321496135) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.39349230643449595) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.053486769241258676) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-3.690138066756159e-06) q[3];
cx q[2],q[3];
rz(-0.1283164034562903) q[0];
rz(-0.08002770953447762) q[1];
rz(-0.20002834888003074) q[2];
rz(-0.1507657301075432) q[3];
rx(-0.14761436396708166) q[0];
rx(-0.07494021900764664) q[1];
rx(-0.2621134949346602) q[2];
rx(0.011730752101029142) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.01679393972403008) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.05384287965658199) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.02524516061590017) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1369229871591798) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.1261290467378879) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.09318080192589917) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.3584017714070062) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1302365735693059) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02362981736396434) q[3];
cx q[2],q[3];
rz(-0.2212714975080057) q[0];
rz(0.004640364094503961) q[1];
rz(-0.21225353734112182) q[2];
rz(-0.039369977941077376) q[3];
rx(-0.11973486365592348) q[0];
rx(-0.023595282530766988) q[1];
rx(-0.09193904755807517) q[2];
rx(-0.0891480890830033) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03480018138174717) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.011741934529609345) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.004808250678167858) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.08503883367800075) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.09877893750312869) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.10905051880326548) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.4102653474427588) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.019508554574215943) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.02590696814289671) q[3];
cx q[2],q[3];
rz(-0.1821317206347467) q[0];
rz(0.016514269441741176) q[1];
rz(-0.08083554381715624) q[2];
rz(0.020313618008187564) q[3];
rx(-0.0639979525284831) q[0];
rx(0.06425582618382707) q[1];
rx(-0.14306891030854554) q[2];
rx(-0.06782497461334609) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.008139520871883662) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0027644102069595746) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.028522998395184417) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.17982310912709765) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07206908159047908) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2254801593344524) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.3964158040286017) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.028232187925280133) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04615698351310946) q[3];
cx q[2],q[3];
rz(-0.2389341902792167) q[0];
rz(-0.05163127388955604) q[1];
rz(0.0025114647046487943) q[2];
rz(0.16992183045077397) q[3];
rx(-0.1559532464749088) q[0];
rx(0.07188834768733539) q[1];
rx(-0.08456101772899607) q[2];
rx(-0.0802279087235431) q[3];