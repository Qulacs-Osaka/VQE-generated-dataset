OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.0612267306424723) q[0];
ry(1.0138521791031583) q[1];
cx q[0],q[1];
ry(-0.696882233641678) q[0];
ry(-1.2947373208467758) q[1];
cx q[0],q[1];
ry(-1.463544593344288) q[1];
ry(2.420509591736178) q[2];
cx q[1],q[2];
ry(2.699507531661422) q[1];
ry(3.102975688364059) q[2];
cx q[1],q[2];
ry(-1.5657493335588026) q[2];
ry(0.10094781655367768) q[3];
cx q[2],q[3];
ry(-0.1275241490195919) q[2];
ry(2.8508867740867148) q[3];
cx q[2],q[3];
ry(-0.9610557526513839) q[3];
ry(0.39117875507622646) q[4];
cx q[3],q[4];
ry(2.7421479327326814) q[3];
ry(2.8570092466958763) q[4];
cx q[3],q[4];
ry(-2.86761810674268) q[4];
ry(2.9028622508974244) q[5];
cx q[4],q[5];
ry(-1.693980126510116) q[4];
ry(-1.7063255878789851) q[5];
cx q[4],q[5];
ry(-0.07397853678347222) q[5];
ry(-0.561034466728195) q[6];
cx q[5],q[6];
ry(-0.49631609245497627) q[5];
ry(0.01274595763876718) q[6];
cx q[5],q[6];
ry(-1.5550176496441448) q[6];
ry(3.0845227899643275) q[7];
cx q[6],q[7];
ry(2.1117944661083126) q[6];
ry(1.9375448334182446) q[7];
cx q[6],q[7];
ry(-0.4632594809856201) q[0];
ry(2.84721379010537) q[1];
cx q[0],q[1];
ry(-0.5728565982413104) q[0];
ry(2.7574491227408338) q[1];
cx q[0],q[1];
ry(2.706713029314574) q[1];
ry(1.761740154581143) q[2];
cx q[1],q[2];
ry(-2.986196722217966) q[1];
ry(0.6601800179553019) q[2];
cx q[1],q[2];
ry(-0.19490821218641496) q[2];
ry(2.577637162299168) q[3];
cx q[2],q[3];
ry(-2.5038118776723315) q[2];
ry(0.13709471926659855) q[3];
cx q[2],q[3];
ry(1.5072972462268357) q[3];
ry(-2.422768454129658) q[4];
cx q[3],q[4];
ry(-0.4613896151007326) q[3];
ry(-2.8943390806865246) q[4];
cx q[3],q[4];
ry(1.5172928422188092) q[4];
ry(-2.689222885382501) q[5];
cx q[4],q[5];
ry(2.9601193773212837) q[4];
ry(2.2730448169395787) q[5];
cx q[4],q[5];
ry(2.9487059233164254) q[5];
ry(1.2768962494709553) q[6];
cx q[5],q[6];
ry(-2.918809722104874) q[5];
ry(0.04612289017134685) q[6];
cx q[5],q[6];
ry(-2.9453724557324015) q[6];
ry(-0.23550249618724273) q[7];
cx q[6],q[7];
ry(-0.16487356820542862) q[6];
ry(2.0342431813693214) q[7];
cx q[6],q[7];
ry(1.075007914132107) q[0];
ry(0.7511409655127977) q[1];
cx q[0],q[1];
ry(-2.4023467337157536) q[0];
ry(0.5624698710115019) q[1];
cx q[0],q[1];
ry(2.405273728501247) q[1];
ry(1.636959469473094) q[2];
cx q[1],q[2];
ry(2.26284991235826) q[1];
ry(-0.7512126160302854) q[2];
cx q[1],q[2];
ry(-2.148787891009887) q[2];
ry(-0.3607581288238254) q[3];
cx q[2],q[3];
ry(0.9017367670856681) q[2];
ry(-2.7883575391349518) q[3];
cx q[2],q[3];
ry(-0.9318671021598743) q[3];
ry(0.7858922349962469) q[4];
cx q[3],q[4];
ry(-1.158242707394124) q[3];
ry(2.194914447174342) q[4];
cx q[3],q[4];
ry(-0.7240525903699978) q[4];
ry(2.19187708680436) q[5];
cx q[4],q[5];
ry(-1.864779409767217) q[4];
ry(0.1978972813895954) q[5];
cx q[4],q[5];
ry(1.4180452241213166) q[5];
ry(1.5637428048221613) q[6];
cx q[5],q[6];
ry(1.6531623353213707) q[5];
ry(0.1644462883117166) q[6];
cx q[5],q[6];
ry(0.23911933486973425) q[6];
ry(1.1692193857525925) q[7];
cx q[6],q[7];
ry(1.368485926922613) q[6];
ry(1.3819939718400163) q[7];
cx q[6],q[7];
ry(-1.2993880534607323) q[0];
ry(-2.1133883600096794) q[1];
cx q[0],q[1];
ry(2.171785221873046) q[0];
ry(-0.38825258895328485) q[1];
cx q[0],q[1];
ry(-1.7532403118280548) q[1];
ry(-0.7659946260619326) q[2];
cx q[1],q[2];
ry(-1.708552354747395) q[1];
ry(1.5062141158715505) q[2];
cx q[1],q[2];
ry(-2.6877381543182888) q[2];
ry(-2.8724758364522183) q[3];
cx q[2],q[3];
ry(0.9816946450261924) q[2];
ry(-1.6100620111747075) q[3];
cx q[2],q[3];
ry(2.8674544340194625) q[3];
ry(1.1350741598447227) q[4];
cx q[3],q[4];
ry(1.6741317567905512) q[3];
ry(-0.08044180454946837) q[4];
cx q[3],q[4];
ry(-1.2987245994248666) q[4];
ry(-1.2067882845833156) q[5];
cx q[4],q[5];
ry(-1.8899176351182159) q[4];
ry(-0.7498907981648628) q[5];
cx q[4],q[5];
ry(0.002869884591837926) q[5];
ry(2.1245291282766523) q[6];
cx q[5],q[6];
ry(-0.4422034298926055) q[5];
ry(0.5260292981406608) q[6];
cx q[5],q[6];
ry(-2.9354584854099106) q[6];
ry(2.174490229333247) q[7];
cx q[6],q[7];
ry(-1.7631167832083419) q[6];
ry(0.0007479752743843724) q[7];
cx q[6],q[7];
ry(0.0107144202309426) q[0];
ry(0.35428005254179207) q[1];
cx q[0],q[1];
ry(-2.9495540728395686) q[0];
ry(-2.360774335208864) q[1];
cx q[0],q[1];
ry(1.7283184865038048) q[1];
ry(-1.23679646906635) q[2];
cx q[1],q[2];
ry(-0.18308707639063485) q[1];
ry(3.1159323157772048) q[2];
cx q[1],q[2];
ry(0.6317820073637065) q[2];
ry(2.5194828882322806) q[3];
cx q[2],q[3];
ry(-0.7338573317879433) q[2];
ry(-1.3678370406507943) q[3];
cx q[2],q[3];
ry(0.5315400076707535) q[3];
ry(1.0753026025889172) q[4];
cx q[3],q[4];
ry(-0.45653550873106497) q[3];
ry(1.0116213099955775) q[4];
cx q[3],q[4];
ry(1.685236528681413) q[4];
ry(-0.5723308450724459) q[5];
cx q[4],q[5];
ry(2.9009623592051628) q[4];
ry(0.3096305336941679) q[5];
cx q[4],q[5];
ry(-0.346032962635749) q[5];
ry(-0.32343895895882396) q[6];
cx q[5],q[6];
ry(0.260449774052309) q[5];
ry(-0.32328461390509045) q[6];
cx q[5],q[6];
ry(-2.08831673322746) q[6];
ry(-0.21675662060061818) q[7];
cx q[6],q[7];
ry(2.904098570682797) q[6];
ry(-1.9925992190467186) q[7];
cx q[6],q[7];
ry(-0.1236882004278752) q[0];
ry(-2.4077883153443107) q[1];
cx q[0],q[1];
ry(0.9941433364326706) q[0];
ry(-1.8892528253435286) q[1];
cx q[0],q[1];
ry(2.113611047120438) q[1];
ry(-2.6722069730350033) q[2];
cx q[1],q[2];
ry(0.5457284688826687) q[1];
ry(-1.782217542420332) q[2];
cx q[1],q[2];
ry(-0.8605583422300743) q[2];
ry(1.6426004127628149) q[3];
cx q[2],q[3];
ry(2.96327852035813) q[2];
ry(-1.723138054583539) q[3];
cx q[2],q[3];
ry(2.1210035948362913) q[3];
ry(0.7391725071299937) q[4];
cx q[3],q[4];
ry(-1.6301729528076325) q[3];
ry(-2.0972070892486787) q[4];
cx q[3],q[4];
ry(-3.107522536402866) q[4];
ry(-2.3528801973745437) q[5];
cx q[4],q[5];
ry(-0.20356640968434297) q[4];
ry(-2.3125778993519175) q[5];
cx q[4],q[5];
ry(-2.7336681064251005) q[5];
ry(-1.7688319550856677) q[6];
cx q[5],q[6];
ry(-0.6901898830528133) q[5];
ry(1.1353452679576232) q[6];
cx q[5],q[6];
ry(-2.897749306100436) q[6];
ry(-0.4695017440489906) q[7];
cx q[6],q[7];
ry(-2.992352738891234) q[6];
ry(1.2250533244371473) q[7];
cx q[6],q[7];
ry(-0.8619585385289598) q[0];
ry(1.4282332816395558) q[1];
cx q[0],q[1];
ry(-1.5403830824049274) q[0];
ry(1.288056274313455) q[1];
cx q[0],q[1];
ry(-3.1285212495831227) q[1];
ry(0.23027689908553547) q[2];
cx q[1],q[2];
ry(2.159501409080006) q[1];
ry(-0.45512624867528656) q[2];
cx q[1],q[2];
ry(2.3924560219432456) q[2];
ry(-2.815547859405759) q[3];
cx q[2],q[3];
ry(-2.212736440529892) q[2];
ry(0.0214322026511633) q[3];
cx q[2],q[3];
ry(-2.474415811085172) q[3];
ry(0.27236951837082923) q[4];
cx q[3],q[4];
ry(1.0983461390951315) q[3];
ry(1.0069342578050477) q[4];
cx q[3],q[4];
ry(2.543856047223658) q[4];
ry(-3.04868293882291) q[5];
cx q[4],q[5];
ry(1.117239414474702) q[4];
ry(0.8030124528260698) q[5];
cx q[4],q[5];
ry(1.9716840831454425) q[5];
ry(-2.160943327911807) q[6];
cx q[5],q[6];
ry(-1.9976593283280437) q[5];
ry(0.21659902248080165) q[6];
cx q[5],q[6];
ry(-1.8095870556149674) q[6];
ry(-1.3727268013738771) q[7];
cx q[6],q[7];
ry(-1.6642446321832116) q[6];
ry(3.1124105655870764) q[7];
cx q[6],q[7];
ry(-2.528141638177514) q[0];
ry(-0.6508359405553863) q[1];
cx q[0],q[1];
ry(-2.4300764117631592) q[0];
ry(2.043860089856392) q[1];
cx q[0],q[1];
ry(-0.14615379301578368) q[1];
ry(-1.2314732628983007) q[2];
cx q[1],q[2];
ry(-2.695846324524452) q[1];
ry(-0.7974155812982914) q[2];
cx q[1],q[2];
ry(-1.8550133982111188) q[2];
ry(1.057791812742221) q[3];
cx q[2],q[3];
ry(0.7037798923289137) q[2];
ry(2.1558878741630636) q[3];
cx q[2],q[3];
ry(1.1774680008191072) q[3];
ry(-2.2044897495409135) q[4];
cx q[3],q[4];
ry(0.6114772563463644) q[3];
ry(-2.122385799015346) q[4];
cx q[3],q[4];
ry(-1.3463027538549694) q[4];
ry(1.8006947440412668) q[5];
cx q[4],q[5];
ry(2.0528091012288465) q[4];
ry(2.4933366075845083) q[5];
cx q[4],q[5];
ry(-2.8106019893032097) q[5];
ry(2.5043490551229115) q[6];
cx q[5],q[6];
ry(1.5278930406025513) q[5];
ry(1.119127270260278) q[6];
cx q[5],q[6];
ry(-1.634542503490962) q[6];
ry(1.504580337798943) q[7];
cx q[6],q[7];
ry(-1.7241877948556492) q[6];
ry(1.7724770962813015) q[7];
cx q[6],q[7];
ry(1.1409608658778545) q[0];
ry(-2.3489612274219365) q[1];
cx q[0],q[1];
ry(-2.4000221174230343) q[0];
ry(-2.7150555774586658) q[1];
cx q[0],q[1];
ry(-0.7984215077443406) q[1];
ry(-0.9178055395785942) q[2];
cx q[1],q[2];
ry(2.9532566254419854) q[1];
ry(1.8036125299709296) q[2];
cx q[1],q[2];
ry(-0.38584076774282394) q[2];
ry(-2.6335275592730674) q[3];
cx q[2],q[3];
ry(-2.575925825370477) q[2];
ry(-0.3668356044044867) q[3];
cx q[2],q[3];
ry(-2.2165733891304424) q[3];
ry(-2.7441512024457966) q[4];
cx q[3],q[4];
ry(0.1657101134350875) q[3];
ry(0.21030332028773094) q[4];
cx q[3],q[4];
ry(1.0636900527454154) q[4];
ry(-2.7291714987765987) q[5];
cx q[4],q[5];
ry(-2.2661903007993867) q[4];
ry(1.4947914701942537) q[5];
cx q[4],q[5];
ry(1.7459420957588279) q[5];
ry(-1.286175165978611) q[6];
cx q[5],q[6];
ry(1.1046340487265063) q[5];
ry(1.5002486079073887) q[6];
cx q[5],q[6];
ry(0.2803145460754939) q[6];
ry(-0.3640545327099822) q[7];
cx q[6],q[7];
ry(0.1170400574471281) q[6];
ry(1.9236869590828916) q[7];
cx q[6],q[7];
ry(0.840581518075477) q[0];
ry(-2.9393983152564007) q[1];
cx q[0],q[1];
ry(-1.20674175653442) q[0];
ry(-1.0963563088332775) q[1];
cx q[0],q[1];
ry(0.38324934581990033) q[1];
ry(1.2136997362794002) q[2];
cx q[1],q[2];
ry(2.7370423321348394) q[1];
ry(-1.5782600263508926) q[2];
cx q[1],q[2];
ry(2.231633416663712) q[2];
ry(-2.80720392561918) q[3];
cx q[2],q[3];
ry(-2.4188922234716568) q[2];
ry(-2.9586156552839826) q[3];
cx q[2],q[3];
ry(1.369704707870957) q[3];
ry(0.20299659600431863) q[4];
cx q[3],q[4];
ry(-1.9751594925256746) q[3];
ry(-0.36464996267940136) q[4];
cx q[3],q[4];
ry(-0.31459593752922643) q[4];
ry(0.6184486889163896) q[5];
cx q[4],q[5];
ry(2.675256260553949) q[4];
ry(1.726578561382226) q[5];
cx q[4],q[5];
ry(1.3533437331028981) q[5];
ry(1.2331115087507867) q[6];
cx q[5],q[6];
ry(0.8040676900700898) q[5];
ry(1.3932131162323254) q[6];
cx q[5],q[6];
ry(-1.157586935912553) q[6];
ry(-2.3438855396114873) q[7];
cx q[6],q[7];
ry(-1.8505324938401997) q[6];
ry(-2.716783622390343) q[7];
cx q[6],q[7];
ry(2.1798483787259055) q[0];
ry(-2.7726904988064454) q[1];
cx q[0],q[1];
ry(-1.9148405505412733) q[0];
ry(-0.37383467210721716) q[1];
cx q[0],q[1];
ry(-1.4086599064891985) q[1];
ry(1.1845930343389872) q[2];
cx q[1],q[2];
ry(0.8684012507041619) q[1];
ry(1.3767969378617542) q[2];
cx q[1],q[2];
ry(-3.091816857953487) q[2];
ry(-0.13458398110592817) q[3];
cx q[2],q[3];
ry(2.5913844255387857) q[2];
ry(0.28171256153713126) q[3];
cx q[2],q[3];
ry(-2.756301283759629) q[3];
ry(2.900305301068383) q[4];
cx q[3],q[4];
ry(-2.7423700191578124) q[3];
ry(1.4314966377763927) q[4];
cx q[3],q[4];
ry(-0.14312316993075586) q[4];
ry(-3.034829121463391) q[5];
cx q[4],q[5];
ry(-1.4280253958343065) q[4];
ry(1.6591229248263595) q[5];
cx q[4],q[5];
ry(0.32913897772839373) q[5];
ry(2.7129765077620283) q[6];
cx q[5],q[6];
ry(-2.464310508419857) q[5];
ry(2.4823111007720335) q[6];
cx q[5],q[6];
ry(-1.3743945703815328) q[6];
ry(0.8604582886633952) q[7];
cx q[6],q[7];
ry(1.3865594626825415) q[6];
ry(-2.2041388671727855) q[7];
cx q[6],q[7];
ry(2.3500903282816576) q[0];
ry(-2.1497473358952437) q[1];
cx q[0],q[1];
ry(2.480855590695014) q[0];
ry(1.7158026759634937) q[1];
cx q[0],q[1];
ry(-2.7490592516942622) q[1];
ry(-0.5525392165895169) q[2];
cx q[1],q[2];
ry(1.821576364901548) q[1];
ry(-1.0420909991439762) q[2];
cx q[1],q[2];
ry(-1.0359196129070123) q[2];
ry(-2.9182245272415814) q[3];
cx q[2],q[3];
ry(0.046321642496162274) q[2];
ry(-0.9144952857074001) q[3];
cx q[2],q[3];
ry(2.7193191580586342) q[3];
ry(2.566370487938812) q[4];
cx q[3],q[4];
ry(2.369219503951853) q[3];
ry(3.0000914484952736) q[4];
cx q[3],q[4];
ry(-0.8817093678818678) q[4];
ry(2.6616194733389453) q[5];
cx q[4],q[5];
ry(1.914279566711051) q[4];
ry(-1.7101506943866245) q[5];
cx q[4],q[5];
ry(2.2517201208370827) q[5];
ry(1.4364716367488466) q[6];
cx q[5],q[6];
ry(-2.956173608446385) q[5];
ry(2.306149740162497) q[6];
cx q[5],q[6];
ry(0.2328946489036749) q[6];
ry(0.5359822935745777) q[7];
cx q[6],q[7];
ry(1.5526285602012715) q[6];
ry(1.1598108913997862) q[7];
cx q[6],q[7];
ry(-1.444493393790169) q[0];
ry(3.0861511144679854) q[1];
cx q[0],q[1];
ry(0.5937567119871044) q[0];
ry(-2.1190687205387206) q[1];
cx q[0],q[1];
ry(0.24577058871021856) q[1];
ry(-0.11739901761776748) q[2];
cx q[1],q[2];
ry(2.5755394843422765) q[1];
ry(0.9151848476333413) q[2];
cx q[1],q[2];
ry(-1.378358485017845) q[2];
ry(-2.3508491067234067) q[3];
cx q[2],q[3];
ry(0.007752479552688345) q[2];
ry(2.7030043964391397) q[3];
cx q[2],q[3];
ry(0.06549667128069014) q[3];
ry(-1.152258978296854) q[4];
cx q[3],q[4];
ry(0.9036798523192845) q[3];
ry(0.59730254655709) q[4];
cx q[3],q[4];
ry(-1.9518998097376226) q[4];
ry(1.212795518145646) q[5];
cx q[4],q[5];
ry(-2.8923729913579743) q[4];
ry(3.050530427979655) q[5];
cx q[4],q[5];
ry(-0.853086894832986) q[5];
ry(1.1429787105300204) q[6];
cx q[5],q[6];
ry(-3.095149779018347) q[5];
ry(0.32416637549036115) q[6];
cx q[5],q[6];
ry(1.111749042294158) q[6];
ry(1.6649947403278684) q[7];
cx q[6],q[7];
ry(0.2478553704953033) q[6];
ry(1.4103132208721925) q[7];
cx q[6],q[7];
ry(1.6761659274010325) q[0];
ry(0.9572869305069636) q[1];
cx q[0],q[1];
ry(0.9430171317445726) q[0];
ry(2.114458262935575) q[1];
cx q[0],q[1];
ry(1.378152025894953) q[1];
ry(-0.15609037468438203) q[2];
cx q[1],q[2];
ry(2.256485867784596) q[1];
ry(-0.18900681763717309) q[2];
cx q[1],q[2];
ry(-1.625079250110109) q[2];
ry(-0.592443212552217) q[3];
cx q[2],q[3];
ry(-0.4053693809843137) q[2];
ry(2.075590387109581) q[3];
cx q[2],q[3];
ry(0.8036011999319115) q[3];
ry(2.744197139400249) q[4];
cx q[3],q[4];
ry(2.921882433701705) q[3];
ry(-0.6502553464037204) q[4];
cx q[3],q[4];
ry(1.3318991647233442) q[4];
ry(0.8764270962995466) q[5];
cx q[4],q[5];
ry(2.571305883760367) q[4];
ry(3.065018456787989) q[5];
cx q[4],q[5];
ry(0.04197456036768177) q[5];
ry(2.5113795003423633) q[6];
cx q[5],q[6];
ry(2.079505756734515) q[5];
ry(1.9165779342741278) q[6];
cx q[5],q[6];
ry(2.5140690079682075) q[6];
ry(3.0533833152643775) q[7];
cx q[6],q[7];
ry(-0.656989781781536) q[6];
ry(-0.14020622758541304) q[7];
cx q[6],q[7];
ry(-1.3053240116507643) q[0];
ry(2.309278824619405) q[1];
cx q[0],q[1];
ry(0.1793303425443984) q[0];
ry(2.099959404498457) q[1];
cx q[0],q[1];
ry(-1.8364454260324514) q[1];
ry(2.113725206280186) q[2];
cx q[1],q[2];
ry(-3.0542411840598036) q[1];
ry(0.26364566007111545) q[2];
cx q[1],q[2];
ry(-1.0583007666867523) q[2];
ry(0.4783399346659899) q[3];
cx q[2],q[3];
ry(-2.9210050134759937) q[2];
ry(-1.9911647347394836) q[3];
cx q[2],q[3];
ry(2.299662035977647) q[3];
ry(-0.23636670953574357) q[4];
cx q[3],q[4];
ry(2.055300458854023) q[3];
ry(1.8408346563222062) q[4];
cx q[3],q[4];
ry(2.2868560544563397) q[4];
ry(0.5365979244955961) q[5];
cx q[4],q[5];
ry(-0.3267574848542993) q[4];
ry(1.6311832462547349) q[5];
cx q[4],q[5];
ry(1.8225993608030473) q[5];
ry(2.1726672485114635) q[6];
cx q[5],q[6];
ry(-1.1601657977702011) q[5];
ry(-2.9317356334145974) q[6];
cx q[5],q[6];
ry(-1.9261683690529594) q[6];
ry(2.123212775134011) q[7];
cx q[6],q[7];
ry(-1.101644927580688) q[6];
ry(2.6308392087639993) q[7];
cx q[6],q[7];
ry(2.313320427956228) q[0];
ry(-2.0134487030630446) q[1];
cx q[0],q[1];
ry(1.9206328511009265) q[0];
ry(-0.127786549065819) q[1];
cx q[0],q[1];
ry(2.7548656000474785) q[1];
ry(2.2738369030447636) q[2];
cx q[1],q[2];
ry(2.1452196570967335) q[1];
ry(-2.7914827498991817) q[2];
cx q[1],q[2];
ry(-3.0607687235874597) q[2];
ry(-2.8082831394874384) q[3];
cx q[2],q[3];
ry(0.5406922754285279) q[2];
ry(-0.0356682512200179) q[3];
cx q[2],q[3];
ry(-0.42402364104189844) q[3];
ry(-1.3495739114815564) q[4];
cx q[3],q[4];
ry(1.8972528844177354) q[3];
ry(-0.6593019305233137) q[4];
cx q[3],q[4];
ry(-0.7767712787040679) q[4];
ry(2.3385349744827746) q[5];
cx q[4],q[5];
ry(2.706075143122649) q[4];
ry(-1.4643963503262383) q[5];
cx q[4],q[5];
ry(-3.017833830400629) q[5];
ry(1.5417275270904103) q[6];
cx q[5],q[6];
ry(-2.4263476238937436) q[5];
ry(2.050474549393225) q[6];
cx q[5],q[6];
ry(1.9679588281843419) q[6];
ry(0.2418825007170522) q[7];
cx q[6],q[7];
ry(2.9919032272317208) q[6];
ry(-1.643400819188464) q[7];
cx q[6],q[7];
ry(-1.89993949988693) q[0];
ry(-2.621632618776427) q[1];
cx q[0],q[1];
ry(2.1544027417881484) q[0];
ry(2.4256047968856547) q[1];
cx q[0],q[1];
ry(1.561699398115672) q[1];
ry(-2.5504062365236906) q[2];
cx q[1],q[2];
ry(-0.17416400186441616) q[1];
ry(2.6873853048823415) q[2];
cx q[1],q[2];
ry(0.7934250705539406) q[2];
ry(-2.1526129677117884) q[3];
cx q[2],q[3];
ry(-0.46963378547524093) q[2];
ry(-0.30689366132234414) q[3];
cx q[2],q[3];
ry(-2.286640963714326) q[3];
ry(-1.4393514622553054) q[4];
cx q[3],q[4];
ry(2.1451096135027266) q[3];
ry(-1.8365448113044418) q[4];
cx q[3],q[4];
ry(-0.5049123744259871) q[4];
ry(2.3980734126138805) q[5];
cx q[4],q[5];
ry(-1.737996633453256) q[4];
ry(2.126317290715633) q[5];
cx q[4],q[5];
ry(-1.9535397640339993) q[5];
ry(-0.16491485410284135) q[6];
cx q[5],q[6];
ry(-1.6679071685380953) q[5];
ry(-2.003665483378764) q[6];
cx q[5],q[6];
ry(-2.1156087973814954) q[6];
ry(-1.6080532344810825) q[7];
cx q[6],q[7];
ry(2.9904704227956858) q[6];
ry(-0.5694619350949139) q[7];
cx q[6],q[7];
ry(0.29947439931445174) q[0];
ry(3.011017169369957) q[1];
cx q[0],q[1];
ry(1.7890643766614942) q[0];
ry(2.5728932504256408) q[1];
cx q[0],q[1];
ry(-3.0410942262980556) q[1];
ry(-0.28236499045287944) q[2];
cx q[1],q[2];
ry(-1.6838996105739303) q[1];
ry(-0.9494236808908156) q[2];
cx q[1],q[2];
ry(-3.014000183997109) q[2];
ry(0.3668249390033864) q[3];
cx q[2],q[3];
ry(0.2308932639964407) q[2];
ry(-1.893414809602918) q[3];
cx q[2],q[3];
ry(-2.1285870619142226) q[3];
ry(-1.6219691240028764) q[4];
cx q[3],q[4];
ry(-2.3249742635762516) q[3];
ry(2.5140327339794957) q[4];
cx q[3],q[4];
ry(-2.3394550808272156) q[4];
ry(-1.0735134706995293) q[5];
cx q[4],q[5];
ry(1.424098781644995) q[4];
ry(-0.37944068424765737) q[5];
cx q[4],q[5];
ry(0.629510633781536) q[5];
ry(-1.360725193955646) q[6];
cx q[5],q[6];
ry(-2.640008008858665) q[5];
ry(1.0604484197574275) q[6];
cx q[5],q[6];
ry(-3.040605947104827) q[6];
ry(-1.568916737427707) q[7];
cx q[6],q[7];
ry(-0.26690053670822245) q[6];
ry(2.5777347558376333) q[7];
cx q[6],q[7];
ry(0.5941074042692652) q[0];
ry(-1.697142498560775) q[1];
cx q[0],q[1];
ry(2.873874270459165) q[0];
ry(-1.6626803280138136) q[1];
cx q[0],q[1];
ry(-1.9943529252018406) q[1];
ry(1.3131421494713322) q[2];
cx q[1],q[2];
ry(-0.20630803581506643) q[1];
ry(-0.47412493774037384) q[2];
cx q[1],q[2];
ry(-2.2554342929640456) q[2];
ry(1.4585457946018012) q[3];
cx q[2],q[3];
ry(0.2913834211733251) q[2];
ry(1.425403650349842) q[3];
cx q[2],q[3];
ry(-2.556074798753789) q[3];
ry(1.3355653178187579) q[4];
cx q[3],q[4];
ry(-2.1848322253320447) q[3];
ry(-2.3288137915133733) q[4];
cx q[3],q[4];
ry(-2.1026464383189802) q[4];
ry(-1.7773664832662939) q[5];
cx q[4],q[5];
ry(0.38157344462794646) q[4];
ry(-0.7431149887291166) q[5];
cx q[4],q[5];
ry(2.7570579193451845) q[5];
ry(0.26376120896128624) q[6];
cx q[5],q[6];
ry(-1.4116900505819105) q[5];
ry(-0.8435466201106178) q[6];
cx q[5],q[6];
ry(-0.6600332456863729) q[6];
ry(-3.0018848296208094) q[7];
cx q[6],q[7];
ry(-0.6936771626878846) q[6];
ry(-1.8151298136984) q[7];
cx q[6],q[7];
ry(-1.7135826029616652) q[0];
ry(-1.6372005618120928) q[1];
cx q[0],q[1];
ry(1.3088528193650584) q[0];
ry(2.5187077232704596) q[1];
cx q[0],q[1];
ry(2.804477029298024) q[1];
ry(1.0373566812105417) q[2];
cx q[1],q[2];
ry(2.9593476683065427) q[1];
ry(-1.3720445195556898) q[2];
cx q[1],q[2];
ry(2.762781476770747) q[2];
ry(0.7373466569062627) q[3];
cx q[2],q[3];
ry(2.46019610461062) q[2];
ry(2.562775020145347) q[3];
cx q[2],q[3];
ry(-0.8014613219393297) q[3];
ry(1.9892309168622244) q[4];
cx q[3],q[4];
ry(1.7287411728803181) q[3];
ry(0.5946100133105051) q[4];
cx q[3],q[4];
ry(0.17982786386645078) q[4];
ry(-1.764548800658333) q[5];
cx q[4],q[5];
ry(-1.075770415762815) q[4];
ry(-0.4342039566310296) q[5];
cx q[4],q[5];
ry(-3.0716727311812373) q[5];
ry(-1.9365044931745505) q[6];
cx q[5],q[6];
ry(-1.2880626531910755) q[5];
ry(1.1468023829847018) q[6];
cx q[5],q[6];
ry(-1.164391703784343) q[6];
ry(1.2253323207830593) q[7];
cx q[6],q[7];
ry(-2.2888610265243625) q[6];
ry(-1.346832467430376) q[7];
cx q[6],q[7];
ry(0.18113344341057333) q[0];
ry(0.10955124165711538) q[1];
cx q[0],q[1];
ry(2.3040325842258) q[0];
ry(1.3843974973491857) q[1];
cx q[0],q[1];
ry(0.09752625473405897) q[1];
ry(-0.36066405683982344) q[2];
cx q[1],q[2];
ry(0.4522743917426235) q[1];
ry(-1.5179651290609868) q[2];
cx q[1],q[2];
ry(-1.574401065033264) q[2];
ry(-0.6301880242514803) q[3];
cx q[2],q[3];
ry(-2.524062917504686) q[2];
ry(-2.7071398805009883) q[3];
cx q[2],q[3];
ry(0.32044295433946335) q[3];
ry(0.5559834867538258) q[4];
cx q[3],q[4];
ry(1.6350256414883644) q[3];
ry(-2.1498866034200734) q[4];
cx q[3],q[4];
ry(-2.4158505138531687) q[4];
ry(-0.07695316993090806) q[5];
cx q[4],q[5];
ry(-0.8511889430360888) q[4];
ry(3.026324291409814) q[5];
cx q[4],q[5];
ry(0.3971367870229123) q[5];
ry(-2.9250152870787502) q[6];
cx q[5],q[6];
ry(3.1285534658366636) q[5];
ry(-2.9732846027632642) q[6];
cx q[5],q[6];
ry(-1.8807954438276235) q[6];
ry(2.9285556223316647) q[7];
cx q[6],q[7];
ry(0.07512643780048435) q[6];
ry(0.8868561699212157) q[7];
cx q[6],q[7];
ry(1.7367270188578017) q[0];
ry(-1.3972264961703136) q[1];
ry(0.1150018813680756) q[2];
ry(-2.9655436491420115) q[3];
ry(-2.9261682147698203) q[4];
ry(-1.2809150716441733) q[5];
ry(-1.1488895440217772) q[6];
ry(-2.1416741964939274) q[7];