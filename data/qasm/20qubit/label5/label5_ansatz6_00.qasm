OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.245256480960017) q[0];
ry(-3.1236011233377683) q[1];
cx q[0],q[1];
ry(-0.2353188559557493) q[0];
ry(-0.3606348749285743) q[1];
cx q[0],q[1];
ry(0.9105564591287951) q[1];
ry(2.598964358819488) q[2];
cx q[1],q[2];
ry(0.20462520489282632) q[1];
ry(1.7844649813375444) q[2];
cx q[1],q[2];
ry(3.054687998210994) q[2];
ry(-1.8674912051182933) q[3];
cx q[2],q[3];
ry(-0.3513369647870983) q[2];
ry(1.1418187615525432) q[3];
cx q[2],q[3];
ry(1.4793075568669467) q[3];
ry(-0.08594219112250734) q[4];
cx q[3],q[4];
ry(-0.5089165742595068) q[3];
ry(3.1170617166213463) q[4];
cx q[3],q[4];
ry(-0.9581338548760104) q[4];
ry(1.0387579264058937) q[5];
cx q[4],q[5];
ry(1.685590198560696) q[4];
ry(-2.2072941945037385) q[5];
cx q[4],q[5];
ry(2.48299826009863) q[5];
ry(1.454424494245189) q[6];
cx q[5],q[6];
ry(1.6861111944895173) q[5];
ry(-1.7968191683704662) q[6];
cx q[5],q[6];
ry(1.4254062028201604) q[6];
ry(2.9500898814082173) q[7];
cx q[6],q[7];
ry(-2.5302126211023426) q[6];
ry(-1.558668683709439) q[7];
cx q[6],q[7];
ry(0.974689448712903) q[7];
ry(2.5205910602484507) q[8];
cx q[7],q[8];
ry(1.795581257820282) q[7];
ry(-2.2576509266367992) q[8];
cx q[7],q[8];
ry(1.863595140341209) q[8];
ry(-3.1403187082085844) q[9];
cx q[8],q[9];
ry(-2.5157714824444826) q[8];
ry(-1.5636760138668255) q[9];
cx q[8],q[9];
ry(-0.799673543650604) q[9];
ry(-0.47665630616356586) q[10];
cx q[9],q[10];
ry(-1.161442051109356) q[9];
ry(-2.1495932289652933) q[10];
cx q[9],q[10];
ry(1.5965362225520607) q[10];
ry(-2.5225959754881564) q[11];
cx q[10],q[11];
ry(-2.17610981897595) q[10];
ry(1.5615846489696343) q[11];
cx q[10],q[11];
ry(-1.3940263881263393) q[11];
ry(-0.8373976094313267) q[12];
cx q[11],q[12];
ry(0.9918569254589389) q[11];
ry(-0.5756516911104441) q[12];
cx q[11],q[12];
ry(-1.862025309900445) q[12];
ry(-2.696563070571483) q[13];
cx q[12],q[13];
ry(0.36562101847717715) q[12];
ry(-1.5644843485743207) q[13];
cx q[12],q[13];
ry(0.14259556592923328) q[13];
ry(-3.1190443984197542) q[14];
cx q[13],q[14];
ry(1.539465734675609) q[13];
ry(-1.4507013330889529) q[14];
cx q[13],q[14];
ry(1.413886337302266) q[14];
ry(2.179185140908317) q[15];
cx q[14],q[15];
ry(1.1906890723843837) q[14];
ry(1.5669136226954006) q[15];
cx q[14],q[15];
ry(0.015876188270382556) q[15];
ry(2.036014243424574) q[16];
cx q[15],q[16];
ry(-1.8143823367852892) q[15];
ry(0.4528594270416525) q[16];
cx q[15],q[16];
ry(-2.4033870525772216) q[16];
ry(-0.07665331720608301) q[17];
cx q[16],q[17];
ry(-1.5264801429182224) q[16];
ry(-1.5580776841296933) q[17];
cx q[16],q[17];
ry(-2.4476933166611086) q[17];
ry(-2.8844629758163096) q[18];
cx q[17],q[18];
ry(2.3397441721675394) q[17];
ry(-0.7274777030374056) q[18];
cx q[17],q[18];
ry(0.8912961053201657) q[18];
ry(2.3403549067268945) q[19];
cx q[18],q[19];
ry(2.464417297491641) q[18];
ry(1.568250351935399) q[19];
cx q[18],q[19];
ry(-2.096487578160459) q[0];
ry(0.3758794816815615) q[1];
cx q[0],q[1];
ry(3.1254959277842094) q[0];
ry(1.1311990216599241) q[1];
cx q[0],q[1];
ry(-2.8694312928427186) q[1];
ry(-0.3117825005782047) q[2];
cx q[1],q[2];
ry(-2.2709396899738428) q[1];
ry(-0.9573924828313976) q[2];
cx q[1],q[2];
ry(1.6559322494445823) q[2];
ry(0.20445280940957195) q[3];
cx q[2],q[3];
ry(1.122149144869229) q[2];
ry(0.5973738804118779) q[3];
cx q[2],q[3];
ry(1.4792317534702526) q[3];
ry(1.5905818748017717) q[4];
cx q[3],q[4];
ry(1.3550070447104952) q[3];
ry(-1.7419902437637063) q[4];
cx q[3],q[4];
ry(-1.5682138383540885) q[4];
ry(-1.5708759889654647) q[5];
cx q[4],q[5];
ry(-1.5472527117786528) q[4];
ry(-0.8611291708962275) q[5];
cx q[4],q[5];
ry(-1.5755173589828848) q[5];
ry(1.5467944519686956) q[6];
cx q[5],q[6];
ry(2.1561161716054134) q[5];
ry(-1.8633689692868314) q[6];
cx q[5],q[6];
ry(-1.60277594653573) q[6];
ry(-1.5668859855729809) q[7];
cx q[6],q[7];
ry(-1.6324978713351457) q[6];
ry(-0.6600962062066675) q[7];
cx q[6],q[7];
ry(-1.51661033469791) q[7];
ry(-1.720509746993819) q[8];
cx q[7],q[8];
ry(-0.18345724555709708) q[7];
ry(-1.4480588035305024) q[8];
cx q[7],q[8];
ry(1.7659029346338702) q[8];
ry(1.5714910378010292) q[9];
cx q[8],q[9];
ry(1.5836993909518027) q[8];
ry(-0.5561996536124075) q[9];
cx q[8],q[9];
ry(-1.543864843223318) q[9];
ry(-1.4686820136357719) q[10];
cx q[9],q[10];
ry(-0.4627559156647868) q[9];
ry(1.6981305987647515) q[10];
cx q[9],q[10];
ry(1.4660335315356408) q[10];
ry(1.5700368964812048) q[11];
cx q[10],q[11];
ry(-1.5675995137625967) q[10];
ry(2.039468842696774) q[11];
cx q[10],q[11];
ry(1.6000937585625943) q[11];
ry(1.5789654776846627) q[12];
cx q[11],q[12];
ry(-1.895529714579868) q[11];
ry(1.7585715933046033) q[12];
cx q[11],q[12];
ry(1.5551280771511562) q[12];
ry(-1.5689372321575723) q[13];
cx q[12],q[13];
ry(-1.5852507654621064) q[12];
ry(-0.5406052902654075) q[13];
cx q[12],q[13];
ry(1.5865496765427236) q[13];
ry(-1.9878743008581718) q[14];
cx q[13],q[14];
ry(-2.8796064086069992) q[13];
ry(1.6262689372086339) q[14];
cx q[13],q[14];
ry(-1.9782482154303584) q[14];
ry(-1.57162763154731) q[15];
cx q[14],q[15];
ry(1.5653413763700088) q[14];
ry(-1.4896442884612187) q[15];
cx q[14],q[15];
ry(1.6374981033109943) q[15];
ry(-1.3959269663991805) q[16];
cx q[15],q[16];
ry(-0.38294998543270126) q[15];
ry(-1.223945684334512) q[16];
cx q[15],q[16];
ry(-1.7429662385540992) q[16];
ry(1.5704497187064925) q[17];
cx q[16],q[17];
ry(-1.5757564214579336) q[16];
ry(-2.515292315927562) q[17];
cx q[16],q[17];
ry(1.6339522873157737) q[17];
ry(1.6559671577739687) q[18];
cx q[17],q[18];
ry(-1.127926526169933) q[17];
ry(-1.497696538586679) q[18];
cx q[17],q[18];
ry(1.4085047082918078) q[18];
ry(-2.4965578187716715) q[19];
cx q[18],q[19];
ry(-1.5655336550183974) q[18];
ry(-3.0018768226224717) q[19];
cx q[18],q[19];
ry(1.7640763552863208) q[0];
ry(2.415329238399539) q[1];
cx q[0],q[1];
ry(2.985839541689972) q[0];
ry(-3.054525371377814) q[1];
cx q[0],q[1];
ry(-2.370698534421699) q[1];
ry(2.160063312138739) q[2];
cx q[1],q[2];
ry(-2.9926485692461706) q[1];
ry(0.952570809719883) q[2];
cx q[1],q[2];
ry(0.9442874697929848) q[2];
ry(-1.5874285539961344) q[3];
cx q[2],q[3];
ry(-2.6636398340595977) q[2];
ry(-0.8758291015331636) q[3];
cx q[2],q[3];
ry(1.5362507171071613) q[3];
ry(-2.8752281002388056) q[4];
cx q[3],q[4];
ry(3.1299558095466655) q[3];
ry(-1.4299348963028382) q[4];
cx q[3],q[4];
ry(-2.8941977805963712) q[4];
ry(-1.5697089158450594) q[5];
cx q[4],q[5];
ry(1.611456646514779) q[4];
ry(-2.633699031539879) q[5];
cx q[4],q[5];
ry(1.5798143480190543) q[5];
ry(-1.5675357495606317) q[6];
cx q[5],q[6];
ry(2.54630079507327) q[5];
ry(-1.8401732548786225) q[6];
cx q[5],q[6];
ry(-1.567135336596338) q[6];
ry(-1.569687284085493) q[7];
cx q[6],q[7];
ry(-1.4965616227562055) q[6];
ry(1.005323051102955) q[7];
cx q[6],q[7];
ry(1.5719804937994448) q[7];
ry(1.5652398222456672) q[8];
cx q[7],q[8];
ry(0.5693624433290134) q[7];
ry(-1.7614668478106845) q[8];
cx q[7],q[8];
ry(1.556365144222843) q[8];
ry(1.5708299110585955) q[9];
cx q[8],q[9];
ry(-1.578541373818316) q[8];
ry(-0.785301383291925) q[9];
cx q[8],q[9];
ry(1.5706798074132733) q[9];
ry(1.5652457467833933) q[10];
cx q[9],q[10];
ry(1.0424352931071426) q[9];
ry(1.4801868409891454) q[10];
cx q[9],q[10];
ry(-1.5650549732215682) q[10];
ry(-1.5715067099322333) q[11];
cx q[10],q[11];
ry(1.5824077515227222) q[10];
ry(-0.7015767395942225) q[11];
cx q[10],q[11];
ry(-1.5650799829280608) q[11];
ry(1.5718321992027056) q[12];
cx q[11],q[12];
ry(0.5186017205965463) q[11];
ry(1.780346839153232) q[12];
cx q[11],q[12];
ry(1.5527240288458792) q[12];
ry(-1.570797532306673) q[13];
cx q[12],q[13];
ry(-1.5892274333945324) q[12];
ry(-1.1464667240461954) q[13];
cx q[12],q[13];
ry(-1.5651228189784687) q[13];
ry(-1.5718168316682695) q[14];
cx q[13],q[14];
ry(1.578359516016016) q[13];
ry(1.5723519857458728) q[14];
cx q[13],q[14];
ry(-1.572927637112115) q[14];
ry(-1.5703615212520898) q[15];
cx q[14],q[15];
ry(1.567201066960742) q[14];
ry(0.37366356237198683) q[15];
cx q[14],q[15];
ry(1.562710088177484) q[15];
ry(-1.5682348753648645) q[16];
cx q[15],q[16];
ry(2.570938445864101) q[15];
ry(1.8815716561247464) q[16];
cx q[15],q[16];
ry(-1.5723656739817535) q[16];
ry(-1.5712927097875298) q[17];
cx q[16],q[17];
ry(-1.567559944456575) q[16];
ry(1.9086012981276488) q[17];
cx q[16],q[17];
ry(1.555329526897078) q[17];
ry(-1.6110677629782442) q[18];
cx q[17],q[18];
ry(-2.9534532954261654) q[17];
ry(1.355157037209025) q[18];
cx q[17],q[18];
ry(-1.5120425333295149) q[18];
ry(-0.6710069783671662) q[19];
cx q[18],q[19];
ry(1.5688686957595857) q[18];
ry(-3.1024947358362605) q[19];
cx q[18],q[19];
ry(-0.08802946615980822) q[0];
ry(-0.6896671212264655) q[1];
ry(1.5555438106135036) q[2];
ry(-1.567623287198817) q[3];
ry(1.5586386038291273) q[4];
ry(1.5737052370064841) q[5];
ry(-1.5590709336983508) q[6];
ry(1.569505050815287) q[7];
ry(-1.5562087301100043) q[8];
ry(1.571078157871411) q[9];
ry(-1.5592658796061392) q[10];
ry(1.57027050154852) q[11];
ry(1.5747040647698842) q[12];
ry(1.570638306905539) q[13];
ry(-1.5745530581522753) q[14];
ry(-1.5731431524142836) q[15];
ry(-1.5934122231142123) q[16];
ry(1.572519285571542) q[17];
ry(-1.5060406204432741) q[18];
ry(3.1412015284429593) q[19];