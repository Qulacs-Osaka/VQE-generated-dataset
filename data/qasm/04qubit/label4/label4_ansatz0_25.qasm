OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09862482242941199) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.029567185344680717) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.10546416187973884) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.007526589851262962) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12140196701404883) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07654256969861875) q[3];
cx q[2],q[3];
rx(-0.08678411964501442) q[0];
rz(-0.11238451151717521) q[0];
rx(-0.04757783752448382) q[1];
rz(-0.0801696266190866) q[1];
rx(-0.15356642449250996) q[2];
rz(-0.07359432678909092) q[2];
rx(-0.014427741164958041) q[3];
rz(-0.07910866983283958) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09270114546736931) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.008972195243474885) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.07895323709645054) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.03191238607541466) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.028156039733773647) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08405325300313883) q[3];
cx q[2],q[3];
rx(-0.06793767343490727) q[0];
rz(-0.08576871353877574) q[0];
rx(-0.06938084008406815) q[1];
rz(-0.08592985156465133) q[1];
rx(-0.1393268179346037) q[2];
rz(-0.12363961145867285) q[2];
rx(-0.025808591585764568) q[3];
rz(-0.043144851740772504) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1728490157631155) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0042226347558114) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0769354263685141) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07910131467697142) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05268180192928173) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0695437815517513) q[3];
cx q[2],q[3];
rx(-0.04550734271012073) q[0];
rz(-0.06647107264344189) q[0];
rx(0.01475376740466973) q[1];
rz(-0.07751098034798831) q[1];
rx(-0.22228948580733623) q[2];
rz(-0.155657824677163) q[2];
rx(-0.00981330840993767) q[3];
rz(-0.03983711108992577) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.21164146915431037) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.026302670007268505) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.09729970913053931) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0005544887550890964) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.10800671027287517) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.017874879285726297) q[3];
cx q[2],q[3];
rx(0.021660091127934652) q[0];
rz(-0.0453274553705032) q[0];
rx(0.014701380348369599) q[1];
rz(-0.08437974418031044) q[1];
rx(-0.14149727214554136) q[2];
rz(-0.11606060682323434) q[2];
rx(-0.025289029045753118) q[3];
rz(-0.02601112426262993) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.22221097479376603) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.03946567059708844) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.09112030073383318) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04862197109430072) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08671355030773202) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.018591335669900785) q[3];
cx q[2],q[3];
rx(0.034090305257061075) q[0];
rz(-0.06797044730025718) q[0];
rx(-0.03142565348748206) q[1];
rz(-0.14449798744389866) q[1];
rx(-0.18307999979878406) q[2];
rz(-0.1079168117974801) q[2];
rx(-0.1257851917242052) q[3];
rz(-0.046180213608965875) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.27732775144722066) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.11327475012672746) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.1270177983564936) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.013929409040466767) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06652419641745237) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03130839141193223) q[3];
cx q[2],q[3];
rx(-0.007868836432075843) q[0];
rz(0.003241354756628346) q[0];
rx(-0.012112010447785017) q[1];
rz(-0.11883641446647918) q[1];
rx(-0.22335797602632732) q[2];
rz(-0.08556242367895012) q[2];
rx(-0.06000488233947352) q[3];
rz(-0.09283483314542831) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.22386795427062967) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.03868326268763749) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05526388973453931) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07982697217539762) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06703154256026202) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.016026137958957604) q[3];
cx q[2],q[3];
rx(0.04590708840715591) q[0];
rz(-0.06291337982853373) q[0];
rx(-0.06710001707305088) q[1];
rz(-0.0904549373638917) q[1];
rx(-0.16570595810722197) q[2];
rz(-0.03834129182324894) q[2];
rx(-0.05565664694042922) q[3];
rz(-0.0780261470451104) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.22944949817782265) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.1114706429943176) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.13659842077970571) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09841522223899231) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0272258493309937) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06345789942042653) q[3];
cx q[2],q[3];
rx(0.03894898417890753) q[0];
rz(-0.008842776182492835) q[0];
rx(-0.0903267940349278) q[1];
rz(-0.0736380502420809) q[1];
rx(-0.1965497781807255) q[2];
rz(-0.08438694917364196) q[2];
rx(-0.07680272235586232) q[3];
rz(0.002430143468070227) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.23593728689290813) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.02762977324708289) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.08670044260555475) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05976317035901658) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0409659940533703) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.178778256525623) q[3];
cx q[2],q[3];
rx(-0.003867644133715159) q[0];
rz(-0.04527053620087365) q[0];
rx(-0.038189045520747734) q[1];
rz(-0.09999825036001117) q[1];
rx(-0.20706707462658075) q[2];
rz(-0.012401573468390003) q[2];
rx(-0.0830915160420297) q[3];
rz(-0.07186572680545535) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.18738702314950925) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.009907075693258644) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.1315570991553869) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12836863058464001) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.00038891454664221725) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.16263802675646022) q[3];
cx q[2],q[3];
rx(-0.04929083773229091) q[0];
rz(-0.0567704920805053) q[0];
rx(-0.08590295282733304) q[1];
rz(-0.08715313699632694) q[1];
rx(-0.2373918729243926) q[2];
rz(-0.05480164855737909) q[2];
rx(-0.13744377588220094) q[3];
rz(-0.11545419799542932) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13759074031920196) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.028183393546726387) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.002103546602671619) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.17041370816223825) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.05051780068478273) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09306435264672043) q[3];
cx q[2],q[3];
rx(-0.030479250492521835) q[0];
rz(0.024926381425034075) q[0];
rx(-0.07396111782738198) q[1];
rz(-0.0987966621237068) q[1];
rx(-0.23049037668489777) q[2];
rz(-0.038054995038120244) q[2];
rx(-0.12944675278688683) q[3];
rz(-0.0948014796089743) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1443460587408179) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.02684769140128258) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.016316060460587034) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2305327498693126) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.10609018992353085) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11454018965012008) q[3];
cx q[2],q[3];
rx(0.016462816544718834) q[0];
rz(0.03187083262233075) q[0];
rx(-0.08892067371161176) q[1];
rz(-0.09264575387656417) q[1];
rx(-0.18214351843343796) q[2];
rz(-0.08276584521713609) q[2];
rx(-0.09046695559590842) q[3];
rz(-0.07795426137402482) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.17682884889180087) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0430947115752073) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.051481443830223954) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.20528985571191488) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.05186291979836864) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0695231365316125) q[3];
cx q[2],q[3];
rx(0.013291699263062699) q[0];
rz(0.025216225884439065) q[0];
rx(-0.09889622373334167) q[1];
rz(-0.07497120029444344) q[1];
rx(-0.17235138077380294) q[2];
rz(-0.06117272019523217) q[2];
rx(-0.11956426079291665) q[3];
rz(-0.029780331080194983) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.19055831332824855) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.01833421509067137) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.01898566807985007) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2303099450782199) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.04817262146837319) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09086866536756026) q[3];
cx q[2],q[3];
rx(-0.045211079747821875) q[0];
rz(0.00697784393307188) q[0];
rx(-0.08132615775783496) q[1];
rz(-0.026266945138348798) q[1];
rx(-0.12879060571865264) q[2];
rz(-0.032787900148422096) q[2];
rx(-0.138668794353255) q[3];
rz(-0.03960456205646695) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13432763904738537) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.05195676920154146) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.02982264080389252) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.30813368243562794) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.07893468226947288) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.10123204802364913) q[3];
cx q[2],q[3];
rx(-0.06664677895663756) q[0];
rz(0.05333449460584348) q[0];
rx(-0.11492039637199275) q[1];
rz(-0.09292102290585134) q[1];
rx(-0.030811866395902252) q[2];
rz(-0.08788044113130118) q[2];
rx(-0.12380776241717042) q[3];
rz(-0.014743960025847429) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.16592539706676088) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.049684436092752186) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.07040249623672035) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.28783974481149416) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.053823687375282134) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07856828522679123) q[3];
cx q[2],q[3];
rx(-0.10104058781026048) q[0];
rz(0.08658682244276757) q[0];
rx(-0.020272702716353154) q[1];
rz(-0.012553218954626764) q[1];
rx(-0.030923262232554444) q[2];
rz(-0.12834307175577483) q[2];
rx(-0.13954277700538997) q[3];
rz(-0.03043211131665366) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.19558925308661396) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.030464231356260637) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.06569211010678834) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.29052152647010027) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.028488805659941507) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.023340919896080687) q[3];
cx q[2],q[3];
rx(-0.08948329541985314) q[0];
rz(0.09281625803952066) q[0];
rx(-0.015162000141822786) q[1];
rz(-0.011909789487586845) q[1];
rx(0.07899434815465928) q[2];
rz(-0.048323367361469964) q[2];
rx(-0.203300600655152) q[3];
rz(-0.07533127142360933) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.21956515192630674) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.028025526856557535) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.10064565204579831) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2623222456401901) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.027107203221393792) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0517458985630376) q[3];
cx q[2],q[3];
rx(-0.14243198158678724) q[0];
rz(0.12118261516989408) q[0];
rx(-0.0788944947812645) q[1];
rz(0.017263856384929187) q[1];
rx(0.04995148499550807) q[2];
rz(-0.06596640300505464) q[2];
rx(-0.2104620744902186) q[3];
rz(0.005782242729238581) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14862256886700717) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.022385552502807382) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.07409129139719116) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.25287291319576544) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.015425907463770814) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.013627445090927915) q[3];
cx q[2],q[3];
rx(-0.1346249038085352) q[0];
rz(0.11549695999964191) q[0];
rx(-0.03276347688846211) q[1];
rz(-0.059586118730409185) q[1];
rx(0.05299201387944853) q[2];
rz(-0.059291475127734694) q[2];
rx(-0.16381102687648882) q[3];
rz(-0.07277476747445201) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1805455180446357) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1073921174298297) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.09149475753054481) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2113720282631993) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.009372861952182891) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0902475637111662) q[3];
cx q[2],q[3];
rx(-0.11634477887116006) q[0];
rz(0.19548725200718045) q[0];
rx(-0.020507181922329434) q[1];
rz(-0.053079495168496235) q[1];
rx(0.014160794084641522) q[2];
rz(-0.11574996577050883) q[2];
rx(-0.16719659026226424) q[3];
rz(-0.007532761609136886) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1779967498359864) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06551871227749734) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.06746888109115054) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.25087447268324525) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04522111483596238) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.015736892425887394) q[3];
cx q[2],q[3];
rx(-0.07293272879809598) q[0];
rz(0.1969421842396535) q[0];
rx(0.009309917069843434) q[1];
rz(0.022337539920473132) q[1];
rx(-0.04289385031039496) q[2];
rz(-0.13019615590715533) q[2];
rx(-0.1770025026241987) q[3];
rz(-0.018355368777189497) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14507852926090545) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10213634029882111) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.07997045079620523) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1752709154616053) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.10276291692224516) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07700614401330742) q[3];
cx q[2],q[3];
rx(-0.1275100604704175) q[0];
rz(0.16701417093766477) q[0];
rx(0.07946407103878707) q[1];
rz(-0.01777510366159611) q[1];
rx(-0.04863196131851988) q[2];
rz(-0.10524223337108322) q[2];
rx(-0.19995116265344043) q[3];
rz(-0.03735560694939613) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.21131951028562346) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1635151389975989) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.13021146847215134) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.16787455674717094) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.184094822394886) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.13155613039872605) q[3];
cx q[2],q[3];
rx(-0.08845216814402723) q[0];
rz(0.12048674425739081) q[0];
rx(0.05100819307853177) q[1];
rz(-0.017198473902762895) q[1];
rx(-0.059067833295104885) q[2];
rz(-0.11677914067152485) q[2];
rx(-0.16530117869648464) q[3];
rz(-0.09130160434929571) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12332811358882012) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1668554936434933) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.05951237005634906) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15137997872977435) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.20453666080705088) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08309923844027871) q[3];
cx q[2],q[3];
rx(-0.1419329822068047) q[0];
rz(0.12698531649215725) q[0];
rx(0.06474637634245879) q[1];
rz(0.003447322337612599) q[1];
rx(-0.059390722112892896) q[2];
rz(-0.12208559230614985) q[2];
rx(-0.1489724335644772) q[3];
rz(-0.11677203928417602) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.16058626150768873) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11384988334721556) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.05106260838049532) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.10078423219445064) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12812046266478425) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.12336969645044374) q[3];
cx q[2],q[3];
rx(-0.07515168687876211) q[0];
rz(0.10396834833844336) q[0];
rx(0.12234129705597599) q[1];
rz(0.029090663495340337) q[1];
rx(-0.02495947567976482) q[2];
rz(-0.06467466089121708) q[2];
rx(-0.12410484226760629) q[3];
rz(-0.08062343392264976) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.17586821507716163) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11600149675860622) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.06486312368357947) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09177628849025737) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1911510093100773) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05573633506183888) q[3];
cx q[2],q[3];
rx(-0.11258765614730006) q[0];
rz(0.07650833965091423) q[0];
rx(0.08746603379304997) q[1];
rz(-0.02678604867944248) q[1];
rx(-0.053306902626892486) q[2];
rz(-0.14426398205701713) q[2];
rx(-0.09720018780582344) q[3];
rz(-0.0925226261524102) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08028143004853408) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09103124186194446) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.02976392123358228) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06632745509560105) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1190648612150055) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.038166390857302994) q[3];
cx q[2],q[3];
rx(-0.06835761319607017) q[0];
rz(0.05560423916883218) q[0];
rx(0.03999961213113) q[1];
rz(-0.004978599588604423) q[1];
rx(0.037304362999027466) q[2];
rz(-0.11980051415961977) q[2];
rx(-0.13943902673859956) q[3];
rz(-0.11251217226810725) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08132259107400998) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.02921501196517913) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.0064949161020489855) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1404517566725852) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08255607343230119) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.008224143222408076) q[3];
cx q[2],q[3];
rx(-0.10456791563789826) q[0];
rz(-0.019461079231232012) q[0];
rx(0.03639860315703111) q[1];
rz(0.01407244242324928) q[1];
rx(-0.014420758960194483) q[2];
rz(-0.11865701154311657) q[2];
rx(-0.12142569838715779) q[3];
rz(-0.07090983236542778) q[3];