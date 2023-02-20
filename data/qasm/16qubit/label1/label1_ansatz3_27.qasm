OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-2.229567711723548) q[0];
rz(-0.061027951011608114) q[0];
ry(2.593666049058354) q[1];
rz(0.37254532625724807) q[1];
ry(-2.5500664897569663) q[2];
rz(-1.1379144481227188) q[2];
ry(1.5328760404462602) q[3];
rz(1.1807624625129545) q[3];
ry(0.5558069107764085) q[4];
rz(-0.9690010486194633) q[4];
ry(3.1388505573144694) q[5];
rz(-1.649284265122758) q[5];
ry(2.1880807491378222) q[6];
rz(-0.47725162426662454) q[6];
ry(2.5001078046800984) q[7];
rz(0.49192129033218635) q[7];
ry(0.006119681931921584) q[8];
rz(0.6405849656526834) q[8];
ry(0.0009927224275912312) q[9];
rz(-0.10886063243907722) q[9];
ry(-1.137582334791874) q[10];
rz(1.9574911641321442) q[10];
ry(2.601327277785665) q[11];
rz(2.894410756377267) q[11];
ry(0.7092847163337863) q[12];
rz(1.8208313726928562) q[12];
ry(-0.8545076128188) q[13];
rz(-1.6331000131793765) q[13];
ry(2.794021086914938) q[14];
rz(2.1762261211007656) q[14];
ry(1.7144715947789733) q[15];
rz(2.5000579182000457) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.3842611162319505) q[0];
rz(-1.5500202788346495) q[0];
ry(1.209480249438805) q[1];
rz(1.418974164639466) q[1];
ry(-1.566991144968549) q[2];
rz(2.9891582491545527) q[2];
ry(-1.4957171724306795) q[3];
rz(0.031561597081656255) q[3];
ry(-0.009575842993610043) q[4];
rz(-0.06700137043290692) q[4];
ry(3.075879926273558) q[5];
rz(-2.4179290411285255) q[5];
ry(0.004346585947697101) q[6];
rz(-2.9839985399937428) q[6];
ry(1.473070241265939) q[7];
rz(0.8393495388707652) q[7];
ry(0.0017046805703409239) q[8];
rz(1.5611850306246358) q[8];
ry(-3.126958810129198) q[9];
rz(-2.7594680748649014) q[9];
ry(2.0544341878730465) q[10];
rz(-0.0362356924229692) q[10];
ry(-0.8923510734597854) q[11];
rz(-0.01588894976370536) q[11];
ry(0.49673405380583807) q[12];
rz(-0.4605511007109898) q[12];
ry(0.32292356607749767) q[13];
rz(-1.9175382305848176) q[13];
ry(-2.4429587175458702) q[14];
rz(-0.7856550165462223) q[14];
ry(-0.6782692293695912) q[15];
rz(-0.9763980072209) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.4492780124164044) q[0];
rz(-0.8661596118817139) q[0];
ry(2.388864272956132) q[1];
rz(2.7122868448243276) q[1];
ry(-0.7776064824753579) q[2];
rz(-1.6065906591355505) q[2];
ry(0.07018483976002354) q[3];
rz(-0.3229477432067815) q[3];
ry(-3.1403840116972526) q[4];
rz(0.4527373877763967) q[4];
ry(0.0026018952666599805) q[5];
rz(1.235355006851346) q[5];
ry(2.4087121526734734) q[6];
rz(1.159209332595776) q[6];
ry(1.303689148208334) q[7];
rz(-0.795346962258388) q[7];
ry(0.0037830769239945994) q[8];
rz(1.2171938563274978) q[8];
ry(0.6504177003922451) q[9];
rz(-0.4556427448804647) q[9];
ry(-2.496797802617536) q[10];
rz(-1.2954263016094878) q[10];
ry(1.5130535288806486) q[11];
rz(0.9027254564519184) q[11];
ry(0.8781489356849024) q[12];
rz(-1.7149110588115395) q[12];
ry(3.0994423615609104) q[13];
rz(-0.9066922701866922) q[13];
ry(2.1841709465031025) q[14];
rz(1.6648142259007586) q[14];
ry(0.7705913340109455) q[15];
rz(-2.3324960076152816) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.5809975261320226) q[0];
rz(2.738276681770279) q[0];
ry(0.8215666223228456) q[1];
rz(-0.44951503496688794) q[1];
ry(2.2841418082646103) q[2];
rz(1.07571830901566) q[2];
ry(-1.6557562779001405) q[3];
rz(2.3100149945640442) q[3];
ry(-3.140838671180724) q[4];
rz(1.9223994520672743) q[4];
ry(-0.29268220084358393) q[5];
rz(-1.4605899903918065) q[5];
ry(-3.113872163827755) q[6];
rz(-2.018364851057399) q[6];
ry(-0.03929407540941465) q[7];
rz(-1.0577965336318613) q[7];
ry(3.1407228701616607) q[8];
rz(-0.9527074061229406) q[8];
ry(-3.128089540995696) q[9];
rz(2.5181795719803546) q[9];
ry(2.5683101551834264) q[10];
rz(2.337501700841205) q[10];
ry(-2.7596992111258825) q[11];
rz(1.5018842764013405) q[11];
ry(-2.3910763670729422) q[12];
rz(1.0855269758250545) q[12];
ry(3.11398359438125) q[13];
rz(-0.18504668358117904) q[13];
ry(-1.1146588470560854) q[14];
rz(1.2284140662534488) q[14];
ry(-1.330131802746366) q[15];
rz(1.5270651532841342) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.7110788181273575) q[0];
rz(3.087246258084214) q[0];
ry(0.07230243265963392) q[1];
rz(-1.3325093070992926) q[1];
ry(3.0836963070693058) q[2];
rz(1.9741582524542054) q[2];
ry(3.083912308740628) q[3];
rz(1.5829558190450002) q[3];
ry(-0.11528287703813422) q[4];
rz(2.73918890619553) q[4];
ry(-3.141478529754327) q[5];
rz(-2.5557919016332096) q[5];
ry(-2.229657639143888) q[6];
rz(2.0303733622516535) q[6];
ry(0.4392980210654054) q[7];
rz(-1.682392658127167) q[7];
ry(3.133524567086044) q[8];
rz(1.1826963047235246) q[8];
ry(1.2998790927333905) q[9];
rz(-2.457711529023593) q[9];
ry(-1.160943102640184) q[10];
rz(2.8510006580892977) q[10];
ry(-1.9540580750327425) q[11];
rz(-2.411053800858331) q[11];
ry(-1.4531923717177655) q[12];
rz(0.9902295481832928) q[12];
ry(0.14362879532274953) q[13];
rz(0.42516882190046257) q[13];
ry(2.1481269659119677) q[14];
rz(-2.5394754426980057) q[14];
ry(2.1355390867076385) q[15];
rz(0.5009309255140417) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.32019286674355) q[0];
rz(0.45251499360461045) q[0];
ry(1.2909656458948255) q[1];
rz(-2.1381183177282264) q[1];
ry(0.17085562394942588) q[2];
rz(1.5316537999557651) q[2];
ry(-0.4131163919912135) q[3];
rz(-2.610949002010476) q[3];
ry(-0.00736306521100083) q[4];
rz(1.082119780234689) q[4];
ry(0.4377768371116642) q[5];
rz(1.3348223634078087) q[5];
ry(3.134508269884626) q[6];
rz(-1.7782452487683003) q[6];
ry(-3.0806089210608034) q[7];
rz(-1.1812440492187544) q[7];
ry(3.1309335949991537) q[8];
rz(0.32197182051944323) q[8];
ry(0.010526235644848551) q[9];
rz(0.8110339288994375) q[9];
ry(-1.4295968639695948) q[10];
rz(0.3739716286705717) q[10];
ry(-1.5652189413460735) q[11];
rz(1.9236512032314046) q[11];
ry(-2.9242345880856986) q[12];
rz(0.6824742269133584) q[12];
ry(-0.2552351734192231) q[13];
rz(-0.8049361178203016) q[13];
ry(-1.9886187713676282) q[14];
rz(0.9277442574710588) q[14];
ry(-0.34609721291244544) q[15];
rz(0.44743150054963543) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.609101567389339) q[0];
rz(2.1218667414395744) q[0];
ry(1.9819531822990768) q[1];
rz(2.6557697267296643) q[1];
ry(0.4284029336232227) q[2];
rz(-0.13781825809098106) q[2];
ry(-0.3639622021767934) q[3];
rz(1.8127923609405752) q[3];
ry(-2.9572178306991077) q[4];
rz(2.3981112406563203) q[4];
ry(0.0003801778925049959) q[5];
rz(-0.5253710683575523) q[5];
ry(2.4350032920928655) q[6];
rz(-0.6798833089101279) q[6];
ry(-2.9741351558658145) q[7];
rz(-2.13764001671922) q[7];
ry(-0.004415523300509276) q[8];
rz(0.8659354155989716) q[8];
ry(2.982477454790109) q[9];
rz(-1.659114076707945) q[9];
ry(-0.9808929226470022) q[10];
rz(-2.7112524732406262) q[10];
ry(-2.151184960323075) q[11];
rz(0.43445926121555656) q[11];
ry(2.6702508784307555) q[12];
rz(-0.19405270237399883) q[12];
ry(-0.37133900516151463) q[13];
rz(2.3252826321843756) q[13];
ry(-0.1538059305955093) q[14];
rz(2.729293557499478) q[14];
ry(2.382876625899826) q[15];
rz(-2.0090672475277156) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.5391739165910915) q[0];
rz(0.4875124321709965) q[0];
ry(2.2479049025136617) q[1];
rz(0.9110736804686915) q[1];
ry(-0.6213678858188487) q[2];
rz(0.12672851210439173) q[2];
ry(-1.3571881668407526) q[3];
rz(0.13125234141522135) q[3];
ry(3.1401471037898574) q[4];
rz(-2.0433454986655155) q[4];
ry(-3.004500276079886) q[5];
rz(-0.23713528952389318) q[5];
ry(0.0198349013686401) q[6];
rz(-0.8178411026981827) q[6];
ry(3.123507284948187) q[7];
rz(1.45565224013645) q[7];
ry(3.1406658812483754) q[8];
rz(2.8785812037763434) q[8];
ry(-0.0031323349399123247) q[9];
rz(2.840300379288301) q[9];
ry(-1.4171320294603156) q[10];
rz(2.021893742924558) q[10];
ry(0.743419765220799) q[11];
rz(-0.9384815656137204) q[11];
ry(-0.5141027040404074) q[12];
rz(1.1922068672183306) q[12];
ry(0.6518301142813261) q[13];
rz(2.443153412521123) q[13];
ry(1.6674828383734486) q[14];
rz(-0.6683679757530134) q[14];
ry(2.747774390792729) q[15];
rz(-0.9769489316277234) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.47814878692578705) q[0];
rz(-2.849205622314623) q[0];
ry(-1.3333816404683843) q[1];
rz(0.661058213749005) q[1];
ry(3.0967259416246415) q[2];
rz(1.79376628281456) q[2];
ry(2.587427318556196) q[3];
rz(0.5507028668280007) q[3];
ry(2.9574187344892877) q[4];
rz(2.6218426889540103) q[4];
ry(-0.003542661316056213) q[5];
rz(2.650199245196199) q[5];
ry(0.23089173095491233) q[6];
rz(1.8276811887613946) q[6];
ry(-3.0136476282189504) q[7];
rz(2.641545906542842) q[7];
ry(0.0008238728875238488) q[8];
rz(-0.3891639685386244) q[8];
ry(2.0852712827404494) q[9];
rz(2.9901708134683798) q[9];
ry(-1.6198001635107389) q[10];
rz(1.3344752164045772) q[10];
ry(0.8731923891021527) q[11];
rz(-1.144984635806935) q[11];
ry(1.052928416579845) q[12];
rz(2.2522939629727765) q[12];
ry(-1.7002178058716186) q[13];
rz(-2.042495360881782) q[13];
ry(2.783502658353961) q[14];
rz(-1.5805789809073376) q[14];
ry(2.6622727163949786) q[15];
rz(2.8291009921578776) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.0981781462362463) q[0];
rz(-0.9782883402576773) q[0];
ry(-0.6861875697831532) q[1];
rz(-2.2922222548906803) q[1];
ry(-0.3501258712362951) q[2];
rz(1.0049516578304658) q[2];
ry(1.0686298804010723) q[3];
rz(-1.9694420129654127) q[3];
ry(-3.1332148273749283) q[4];
rz(2.657217404649845) q[4];
ry(3.051883627239167) q[5];
rz(-1.0776193380110497) q[5];
ry(-0.004522774802966759) q[6];
rz(0.9644445707055898) q[6];
ry(-1.99801836384027) q[7];
rz(-1.106930443044324) q[7];
ry(1.4360794499501306) q[8];
rz(1.2676338951252208) q[8];
ry(-0.010138811318095796) q[9];
rz(0.48976504176193486) q[9];
ry(-2.1639797744513265) q[10];
rz(-2.104760099426179) q[10];
ry(-0.2412069221180451) q[11];
rz(1.2566133218858004) q[11];
ry(-0.7452298117859026) q[12];
rz(-1.0825843046745192) q[12];
ry(-1.7118873693011034) q[13];
rz(3.1135919020884297) q[13];
ry(-0.00014867007504193452) q[14];
rz(2.9181085390627484) q[14];
ry(2.5249395266231534) q[15];
rz(0.6444824860902978) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.818346156283917) q[0];
rz(2.390235405307145) q[0];
ry(-1.642819830334241) q[1];
rz(-0.24757316817304442) q[1];
ry(-1.7889196724828875) q[2];
rz(1.1996269455714121) q[2];
ry(-0.8475687291978843) q[3];
rz(-2.01109567964763) q[3];
ry(-0.5165026740396896) q[4];
rz(1.392189986780898) q[4];
ry(-0.0025766703299812126) q[5];
rz(1.7200072030255404) q[5];
ry(-1.5066761863112705) q[6];
rz(-0.6071461966361512) q[6];
ry(-2.9949023548322513) q[7];
rz(-2.63806212450908) q[7];
ry(-3.117753404336169) q[8];
rz(1.2487365930634837) q[8];
ry(0.016324340267692406) q[9];
rz(0.7989396513029856) q[9];
ry(-3.136279645663779) q[10];
rz(-1.3418336478091426) q[10];
ry(-1.7603969278865164) q[11];
rz(1.055824309795751) q[11];
ry(-2.045488349954198) q[12];
rz(2.2414401966777966) q[12];
ry(-1.5089577001488523) q[13];
rz(1.0886637766596114) q[13];
ry(-1.1867269001438816) q[14];
rz(0.29084086221347943) q[14];
ry(-2.575646354978618) q[15];
rz(1.7543878657956231) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.1666341094167532) q[0];
rz(0.8582782542488082) q[0];
ry(0.8769693912107464) q[1];
rz(2.4602693326103497) q[1];
ry(-0.6122429152760585) q[2];
rz(1.4117130789444552) q[2];
ry(1.575485773773432) q[3];
rz(0.560226349486542) q[3];
ry(-0.0005847731542424052) q[4];
rz(2.4204447684402632) q[4];
ry(3.096826751629554) q[5];
rz(0.580898672464716) q[5];
ry(-0.0018293461427879981) q[6];
rz(-2.295529446794379) q[6];
ry(1.865282547724009) q[7];
rz(-1.2717751674282456) q[7];
ry(1.4617417264641377) q[8];
rz(2.4429386638341035) q[8];
ry(3.136712574491678) q[9];
rz(-0.5062600419212231) q[9];
ry(0.9181871532975397) q[10];
rz(-0.43059698619750336) q[10];
ry(2.8712581848481804) q[11];
rz(0.450151982804113) q[11];
ry(2.3657968312769686) q[12];
rz(0.7335670208138105) q[12];
ry(2.3262487135626344) q[13];
rz(3.1284397588272843) q[13];
ry(0.6593483878416208) q[14];
rz(-1.2604959759882801) q[14];
ry(0.8920236428342161) q[15];
rz(1.8913467012741378) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.23419746204253844) q[0];
rz(-2.4483925337235974) q[0];
ry(-2.3903001669287485) q[1];
rz(2.5418774840488716) q[1];
ry(-0.6667272632244013) q[2];
rz(2.0632987595701584) q[2];
ry(1.2160343565102583) q[3];
rz(-1.5088099626488312) q[3];
ry(1.621474683898201) q[4];
rz(2.683318457122352) q[4];
ry(3.1378522930639483) q[5];
rz(-1.7622065137359877) q[5];
ry(2.553110468929689) q[6];
rz(-1.7873075770915294) q[6];
ry(0.08785167898431556) q[7];
rz(0.912214550039022) q[7];
ry(0.010044687938868258) q[8];
rz(1.8866708246839186) q[8];
ry(-0.013032571159215856) q[9];
rz(-1.0479966678332753) q[9];
ry(-0.00020508620102965838) q[10];
rz(-1.5026343372244169) q[10];
ry(0.620460404099747) q[11];
rz(-1.723044578525838) q[11];
ry(2.6164388136097214) q[12];
rz(0.0965004654320456) q[12];
ry(1.2981730321980494) q[13];
rz(2.1100877206696405) q[13];
ry(2.482285152987232) q[14];
rz(0.30679725566536936) q[14];
ry(2.0892551682881146) q[15];
rz(0.05949195050351353) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(2.072254305366597) q[0];
rz(2.7271269884581546) q[0];
ry(1.8360592532276012) q[1];
rz(1.6753914512643933) q[1];
ry(-0.9557556980354264) q[2];
rz(1.165732574754109) q[2];
ry(-2.050405359710855) q[3];
rz(3.117321525493485) q[3];
ry(-0.00709540109137663) q[4];
rz(2.3180057054342766) q[4];
ry(-3.1336615915001067) q[5];
rz(-0.5838132292795145) q[5];
ry(-3.1414151936806283) q[6];
rz(3.0914173945585692) q[6];
ry(-1.8625619223886583) q[7];
rz(-1.7477998892295163) q[7];
ry(-1.4765518868426581) q[8];
rz(-2.2246659064276706) q[8];
ry(-3.1391802562291664) q[9];
rz(-1.5664266079623816) q[9];
ry(0.1462648856833366) q[10];
rz(2.2680807121814963) q[10];
ry(1.4477811571474304) q[11];
rz(2.86111533962087) q[11];
ry(1.9750303313947908) q[12];
rz(2.0608008846807335) q[12];
ry(0.6037505310016554) q[13];
rz(-2.725815846897128) q[13];
ry(-1.7780513636962636) q[14];
rz(-1.0976502110626556) q[14];
ry(-0.6709309008660718) q[15];
rz(-0.11706954153901067) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.38359592426950273) q[0];
rz(-0.925201358512639) q[0];
ry(0.49427117774903007) q[1];
rz(-2.0333517816697446) q[1];
ry(-1.9767586394831103) q[2];
rz(-2.7658954960552875) q[2];
ry(-1.2486420265005327) q[3];
rz(0.38034189294048887) q[3];
ry(-1.5748343897405608) q[4];
rz(2.215005304501079) q[4];
ry(-3.1296912934971157) q[5];
rz(2.9136205967040887) q[5];
ry(-1.3325387570441363) q[6];
rz(-2.2896048366122073) q[6];
ry(1.178039923148555) q[7];
rz(-2.989880713792891) q[7];
ry(-3.1145816533305655) q[8];
rz(-0.4342767485701406) q[8];
ry(0.0012123004608397229) q[9];
rz(2.3078251157597145) q[9];
ry(2.3941074110630645) q[10];
rz(2.903814108198918) q[10];
ry(-1.2699579452430194) q[11];
rz(-2.329929260529968) q[11];
ry(1.2594493814700147) q[12];
rz(-0.9545355218221633) q[12];
ry(0.4085627035164272) q[13];
rz(-0.6248667032445011) q[13];
ry(-2.5095547194620083) q[14];
rz(-1.698168555301868) q[14];
ry(1.2624134029787066) q[15];
rz(0.7446207992614267) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-3.0367959077069027) q[0];
rz(0.39256263227242894) q[0];
ry(1.3438937237361168) q[1];
rz(1.99965445107292) q[1];
ry(0.8320750004720522) q[2];
rz(-0.3928347232999237) q[2];
ry(3.002535531667185) q[3];
rz(1.1451314394960195) q[3];
ry(0.001197336903223967) q[4];
rz(0.052375535248502074) q[4];
ry(-0.006159141280455138) q[5];
rz(-1.1710970176224536) q[5];
ry(0.009060781240708037) q[6];
rz(-0.31501846690507856) q[6];
ry(-2.1313449282950163) q[7];
rz(0.6323001666837005) q[7];
ry(-0.004391396080770882) q[8];
rz(-0.8911458356419599) q[8];
ry(3.130454424253178) q[9];
rz(2.7862979902302167) q[9];
ry(0.564470936206397) q[10];
rz(-2.043978461063918) q[10];
ry(1.8902546469647525) q[11];
rz(-0.4057073960442416) q[11];
ry(0.6751924453281183) q[12];
rz(-1.8684652281185965) q[12];
ry(1.1630180120496876) q[13];
rz(-1.8031603858262484) q[13];
ry(2.18748367016492) q[14];
rz(-1.77372249105027) q[14];
ry(-1.9353051931623695) q[15];
rz(-0.7609771577038753) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.9741488651922883) q[0];
rz(2.4784657508495838) q[0];
ry(1.1938487793979906) q[1];
rz(-0.2222785809702712) q[1];
ry(1.7533028010353737) q[2];
rz(0.29624152689016175) q[2];
ry(3.082998102683196) q[3];
rz(0.6899090322419358) q[3];
ry(2.3330885571058593) q[4];
rz(-1.6364414700608323) q[4];
ry(-1.5799058135218313) q[5];
rz(-0.7815557319195713) q[5];
ry(-0.411496605831851) q[6];
rz(-0.7884106948852239) q[6];
ry(-2.2242278937041347) q[7];
rz(1.2540561508893724) q[7];
ry(2.825196866430481) q[8];
rz(-2.5626518993853464) q[8];
ry(3.0953951591898914) q[9];
rz(-0.13597016022974764) q[9];
ry(-2.2560622955024625) q[10];
rz(-1.8030545767194188) q[10];
ry(-2.9896825226808432) q[11];
rz(-0.26772322823838834) q[11];
ry(-2.279556643288007) q[12];
rz(2.167193399729172) q[12];
ry(2.315556893223882) q[13];
rz(-1.6436552354110723) q[13];
ry(1.963481478530194) q[14];
rz(-1.053210990773433) q[14];
ry(2.9811913083509114) q[15];
rz(0.9611116064103671) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(3.034458575424359) q[0];
rz(-1.8977108046512692) q[0];
ry(1.334160470926365) q[1];
rz(-1.8165588332604017) q[1];
ry(1.9707996784527495) q[2];
rz(-2.75571715774611) q[2];
ry(0.0012107905565414967) q[3];
rz(-2.562572177739201) q[3];
ry(-0.008906247807753509) q[4];
rz(0.34742430458115775) q[4];
ry(3.1392567872349697) q[5];
rz(-2.8349057796536306) q[5];
ry(0.01866887462752942) q[6];
rz(0.28701454117641134) q[6];
ry(-1.604868862291144) q[7];
rz(0.43068935472778413) q[7];
ry(-0.021589198573164634) q[8];
rz(-0.5231295130636493) q[8];
ry(0.00968368027160249) q[9];
rz(1.9665333630207218) q[9];
ry(3.068023639742143) q[10];
rz(-1.203475411222561) q[10];
ry(-1.5839845822740557) q[11];
rz(-1.7955551804821415) q[11];
ry(2.6747943304516375) q[12];
rz(2.163392350132075) q[12];
ry(1.3141932661573463) q[13];
rz(2.134847041584613) q[13];
ry(1.0606640426853895) q[14];
rz(0.5012210258309358) q[14];
ry(-1.4272879912229124) q[15];
rz(-3.00430228004588) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-2.764796905043738) q[0];
rz(-0.8953576983755678) q[0];
ry(-2.6039197366575078) q[1];
rz(1.708401547174368) q[1];
ry(-1.5204121520624312) q[2];
rz(0.5576868651880771) q[2];
ry(2.1040812117951537) q[3];
rz(-1.256894907617963) q[3];
ry(3.055343864692565) q[4];
rz(0.4985422290695378) q[4];
ry(-0.09264744007000658) q[5];
rz(-2.303666528841387) q[5];
ry(-1.543380193332027) q[6];
rz(-2.1276943272232645) q[6];
ry(2.5307838248531502) q[7];
rz(1.8804073941085995) q[7];
ry(0.213997064204275) q[8];
rz(-0.5967111802177181) q[8];
ry(0.0014080048187414818) q[9];
rz(1.3544952035281703) q[9];
ry(-1.3690594699361178) q[10];
rz(0.22804865202020724) q[10];
ry(0.4773096443439133) q[11];
rz(-1.2162463128966297) q[11];
ry(-2.5174561988136) q[12];
rz(-2.904655029338953) q[12];
ry(-2.5733272737764405) q[13];
rz(2.6932555419268294) q[13];
ry(0.3316356026637628) q[14];
rz(0.9544043141852318) q[14];
ry(0.5331378774272366) q[15];
rz(1.9639514710763457) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.2016219687265381) q[0];
rz(3.1013880451714737) q[0];
ry(0.23896399103748042) q[1];
rz(0.3145596809293467) q[1];
ry(1.74852251926185) q[2];
rz(-1.8565713392808108) q[2];
ry(-3.1350028642047385) q[3];
rz(-1.7869957422435243) q[3];
ry(-3.1379438698388347) q[4];
rz(-1.8235393242309796) q[4];
ry(0.0003796663574391346) q[5];
rz(-0.3615719192797696) q[5];
ry(0.016588864393891366) q[6];
rz(-2.648162361313143) q[6];
ry(-1.5617500777137112) q[7];
rz(-1.808594771465765) q[7];
ry(-3.1407014157112836) q[8];
rz(2.8585828443752916) q[8];
ry(0.01448771393663151) q[9];
rz(-1.9387757435676918) q[9];
ry(2.9788316963552557) q[10];
rz(-2.2572508486041194) q[10];
ry(-2.600934373604185) q[11];
rz(-1.6732649269638147) q[11];
ry(-2.9471813981829618) q[12];
rz(-1.6312072868871281) q[12];
ry(-1.1286928734146182) q[13];
rz(-2.5762663675734734) q[13];
ry(-0.9160217296677136) q[14];
rz(-2.7901253494019667) q[14];
ry(1.760263820518279) q[15];
rz(1.5222237067186537) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.09787186337666502) q[0];
rz(-2.4521024155174733) q[0];
ry(-0.05137238063686378) q[1];
rz(-1.3626551880849498) q[1];
ry(2.9162031621679003) q[2];
rz(-0.753795793626983) q[2];
ry(-2.3610240240706273) q[3];
rz(-0.022950722398562277) q[3];
ry(-0.031694541306693344) q[4];
rz(0.10870346100625819) q[4];
ry(1.5354110849596534) q[5];
rz(1.4392456112177259) q[5];
ry(-1.4061726230743812) q[6];
rz(-2.628204219140219) q[6];
ry(-0.1319894820283813) q[7];
rz(-0.36567790049390675) q[7];
ry(-0.6207602870414535) q[8];
rz(1.7225985978674252) q[8];
ry(-3.133893245285815) q[9];
rz(1.0850623037626055) q[9];
ry(-1.6181839503841813) q[10];
rz(2.3476175095386433) q[10];
ry(-0.2792967077846802) q[11];
rz(1.622892810307103) q[11];
ry(2.5039896930477905) q[12];
rz(2.7789967191115474) q[12];
ry(-2.7622806124311947) q[13];
rz(2.7107021074322266) q[13];
ry(-1.5905254542033542) q[14];
rz(-1.8866505269365639) q[14];
ry(-1.3778532707666482) q[15];
rz(2.65224574486004) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-3.1171474517604323) q[0];
rz(1.6070395405927949) q[0];
ry(1.4314972206558274) q[1];
rz(-0.056814003916281025) q[1];
ry(-2.797753835413733) q[2];
rz(-2.5532787691899337) q[2];
ry(0.7603859436506127) q[3];
rz(2.011818147409234) q[3];
ry(3.1367732937197044) q[4];
rz(-0.42897550464146933) q[4];
ry(3.134576455653943) q[5];
rz(0.3478632646732445) q[5];
ry(3.127551063840891) q[6];
rz(2.7306622739589868) q[6];
ry(-3.0686692344611264) q[7];
rz(-0.5129208248985603) q[7];
ry(-0.04175136258547951) q[8];
rz(1.1771255404456182) q[8];
ry(0.01750653052126277) q[9];
rz(0.7312907979000752) q[9];
ry(-0.08378756426691401) q[10];
rz(2.812096446016641) q[10];
ry(1.5549404785032714) q[11];
rz(-0.5583408215166985) q[11];
ry(1.2475999709449113) q[12];
rz(1.4048132504572164) q[12];
ry(-2.8681819595748594) q[13];
rz(1.3167189257478604) q[13];
ry(-3.0606135622488204) q[14];
rz(2.735483421032878) q[14];
ry(-1.6094224373366133) q[15];
rz(-0.586267909344544) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.6587561760232032) q[0];
rz(-2.490053219436743) q[0];
ry(0.29116608900675184) q[1];
rz(-1.0664147575365908) q[1];
ry(2.3277762559046247) q[2];
rz(-0.20145413497209066) q[2];
ry(0.20632917939977016) q[3];
rz(-0.5859292789039766) q[3];
ry(-1.637069990301268) q[4];
rz(-0.4115855689985253) q[4];
ry(-0.02718321184126982) q[5];
rz(-1.9301223323173673) q[5];
ry(-1.7207437124060903) q[6];
rz(-3.0701557665636234) q[6];
ry(-1.503151477161831) q[7];
rz(-1.2656751946328615) q[7];
ry(-2.020938469921393) q[8];
rz(-1.3333323302905076) q[8];
ry(3.131916994416971) q[9];
rz(0.8252738772185549) q[9];
ry(-0.1832374857875796) q[10];
rz(-2.629966520171834) q[10];
ry(0.6499061300571674) q[11];
rz(1.42042205970638) q[11];
ry(2.24799957465706) q[12];
rz(-1.1671594061462836) q[12];
ry(-1.5768865184143763) q[13];
rz(-0.5252958699749791) q[13];
ry(-2.1644729567015704) q[14];
rz(2.808388041146959) q[14];
ry(-2.291798416576663) q[15];
rz(-1.5921185666438298) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-2.632354933566825) q[0];
rz(1.2503669189857094) q[0];
ry(2.741519446808548) q[1];
rz(-2.7853006397267186) q[1];
ry(0.7059844968675623) q[2];
rz(-0.2093977127268323) q[2];
ry(-2.9853192475289685) q[3];
rz(-1.6102251543521264) q[3];
ry(-3.139944121283428) q[4];
rz(-0.06286170494074862) q[4];
ry(-3.141537410422495) q[5];
rz(-2.0472344908429063) q[5];
ry(0.0019559330256528895) q[6];
rz(1.7397521388751) q[6];
ry(-1.7385637623504187) q[7];
rz(-1.228061835156626) q[7];
ry(-0.015430446957043448) q[8];
rz(-2.8206736953552505) q[8];
ry(3.1409285246061267) q[9];
rz(-1.1437432340164975) q[9];
ry(-0.05965218563588958) q[10];
rz(0.15048107165239463) q[10];
ry(-2.2022136955095593) q[11];
rz(1.485938986650236) q[11];
ry(-2.137421811918005) q[12];
rz(0.9688477808883964) q[12];
ry(-1.7382611203167935) q[13];
rz(-0.7991769779107597) q[13];
ry(1.0237613009959723) q[14];
rz(-1.527851229879098) q[14];
ry(2.198663247166402) q[15];
rz(-0.03737189973207004) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.3528998908255776) q[0];
rz(-0.16885487835857305) q[0];
ry(-1.6098381732469882) q[1];
rz(1.3456778343612508) q[1];
ry(-2.1521745424375425) q[2];
rz(0.5857635130923476) q[2];
ry(-0.6623885353369943) q[3];
rz(1.8059069427258638) q[3];
ry(1.6290168379194978) q[4];
rz(-0.3473158573379637) q[4];
ry(-0.2105664579119688) q[5];
rz(-0.33100268010084655) q[5];
ry(2.979230626069046) q[6];
rz(-1.4182892719690816) q[6];
ry(-1.6067426247430836) q[7];
rz(1.3539889767154498) q[7];
ry(-1.6248616277217858) q[8];
rz(0.5133043272054785) q[8];
ry(0.002859782010472775) q[9];
rz(-1.7219160526374369) q[9];
ry(-2.9736109717429673) q[10];
rz(-2.101785257733293) q[10];
ry(0.6501497482981456) q[11];
rz(-1.7651159349812016) q[11];
ry(-1.6004319085136356) q[12];
rz(-0.22391510962368422) q[12];
ry(0.6836427128432865) q[13];
rz(-0.7451262036504643) q[13];
ry(2.4964428523925566) q[14];
rz(-1.5129630864778125) q[14];
ry(-2.28455918375365) q[15];
rz(-1.9805712476055266) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.472323026139056) q[0];
rz(-1.984137353156754) q[0];
ry(-1.3148853736700703) q[1];
rz(2.961519400378339) q[1];
ry(-1.4310521794089315) q[2];
rz(-1.1474760272389564) q[2];
ry(2.9242180019708037) q[3];
rz(0.2912573615011685) q[3];
ry(3.141001607436703) q[4];
rz(-0.6752684578496229) q[4];
ry(0.003842940814896775) q[5];
rz(2.2059179159703715) q[5];
ry(3.1391369451730466) q[6];
rz(-1.7443024384526813) q[6];
ry(1.649679261237378) q[7];
rz(-1.8019030312011672) q[7];
ry(-0.008508802427078004) q[8];
rz(-0.9272476762572489) q[8];
ry(-3.13060350514557) q[9];
rz(1.0166019742335166) q[9];
ry(0.016235710616513663) q[10];
rz(0.6607005968547243) q[10];
ry(-1.540577587154072) q[11];
rz(2.5891151356656215) q[11];
ry(1.332275904948977) q[12];
rz(0.14947044780677224) q[12];
ry(-0.9036974133485028) q[13];
rz(-0.9375077883400557) q[13];
ry(-1.3484722272017011) q[14];
rz(1.883083821771387) q[14];
ry(-0.7272791181663782) q[15];
rz(-0.8888555729608739) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.416234551076279) q[0];
rz(0.5282948214391663) q[0];
ry(-0.12133559091718417) q[1];
rz(-1.4557150588113423) q[1];
ry(0.1233184489922132) q[2];
rz(-2.037699375604337) q[2];
ry(1.560258432262125) q[3];
rz(2.985419183044848) q[3];
ry(-2.8796383913486685) q[4];
rz(0.7317640255616702) q[4];
ry(-3.1193917969504437) q[5];
rz(2.5497604027545444) q[5];
ry(-1.54419985176024) q[6];
rz(-1.6610143028245992) q[6];
ry(-1.0441564020904273) q[7];
rz(0.4977915371118478) q[7];
ry(1.6352429216479507) q[8];
rz(-0.1466493146385721) q[8];
ry(-3.137007504721852) q[9];
rz(0.6026120836251363) q[9];
ry(2.7600567002599554) q[10];
rz(1.8901968463216932) q[10];
ry(-1.0485211462380777) q[11];
rz(1.1617505501094167) q[11];
ry(2.8671212272090805) q[12];
rz(1.2127892785845116) q[12];
ry(-2.3005006276037734) q[13];
rz(0.5329978796527053) q[13];
ry(0.6415616316757188) q[14];
rz(-0.5853523757674988) q[14];
ry(2.315818669920714) q[15];
rz(-1.024373983479777) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.1066853581665792) q[0];
rz(-2.221023235628456) q[0];
ry(-2.2517170474856076) q[1];
rz(2.4961591300323236) q[1];
ry(-2.125373032526251) q[2];
rz(2.8397529124426035) q[2];
ry(-2.9537930223813063) q[3];
rz(1.958839900836768) q[3];
ry(0.010270794058098609) q[4];
rz(-0.6472141857302612) q[4];
ry(0.021901489897385495) q[5];
rz(0.8372462896504822) q[5];
ry(0.019408818987188603) q[6];
rz(2.8643798076047187) q[6];
ry(-1.5558796269726045) q[7];
rz(-2.1786623473387494) q[7];
ry(-0.06798337086383621) q[8];
rz(-0.04577303294514452) q[8];
ry(-3.1180082385754337) q[9];
rz(-2.2953189863110888) q[9];
ry(0.09370197202904905) q[10];
rz(1.1559719352628761) q[10];
ry(2.1884432863208443) q[11];
rz(-1.297170678024723) q[11];
ry(0.7333424188414505) q[12];
rz(0.6263759739585631) q[12];
ry(1.2628919767194207) q[13];
rz(1.4318728726859817) q[13];
ry(-0.5428812948953006) q[14];
rz(-2.7760499396236415) q[14];
ry(2.718677088041791) q[15];
rz(-2.3513541410771026) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.434221650755954) q[0];
rz(0.3207817967385295) q[0];
ry(1.9142613645704394) q[1];
rz(-0.9289809921791) q[1];
ry(-1.8173481133978422) q[2];
rz(-0.09636081752065717) q[2];
ry(-0.22954272587128546) q[3];
rz(0.26970799646469334) q[3];
ry(-2.9107069928626546) q[4];
rz(2.435876303416987) q[4];
ry(1.894512729752338) q[5];
rz(-0.43117664626460606) q[5];
ry(-0.7566775651126072) q[6];
rz(-2.0695738984233687) q[6];
ry(2.715261651696996) q[7];
rz(0.9726887950386071) q[7];
ry(-2.4290883972689667) q[8];
rz(-2.005779720073906) q[8];
ry(3.0938856842224047) q[9];
rz(3.030350559469773) q[9];
ry(-2.9530417429556852) q[10];
rz(-0.8377518310121722) q[10];
ry(1.0955021564374738) q[11];
rz(1.4421169720703313) q[11];
ry(2.0229570958362504) q[12];
rz(3.084410358572665) q[12];
ry(-1.4939678639317462) q[13];
rz(-0.8963816506009693) q[13];
ry(-1.2702649768662932) q[14];
rz(2.3260676821354096) q[14];
ry(-0.02246252904043257) q[15];
rz(2.139590302041677) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.9323452663210682) q[0];
rz(0.7565604852673455) q[0];
ry(1.562667187703412) q[1];
rz(-2.363599111327763) q[1];
ry(-1.6285125053520217) q[2];
rz(2.642718126655259) q[2];
ry(-0.0046915847060295945) q[3];
rz(-1.0076540861972376) q[3];
ry(3.1297579363949937) q[4];
rz(-2.4735449579499953) q[4];
ry(-0.02101625532327872) q[5];
rz(2.098466926401156) q[5];
ry(0.01775464477233268) q[6];
rz(-0.016863732425133036) q[6];
ry(3.1191365561777546) q[7];
rz(1.8747771053971516) q[7];
ry(3.082496263871477) q[8];
rz(1.0777100004285303) q[8];
ry(0.024726380510473156) q[9];
rz(-1.2167796408957923) q[9];
ry(0.06897576492736501) q[10];
rz(-1.1741917657853636) q[10];
ry(-2.2471851761737707) q[11];
rz(-2.36397687620648) q[11];
ry(-1.172547998236829) q[12];
rz(2.9144222502394457) q[12];
ry(2.6425231421739586) q[13];
rz(-1.7636091393454247) q[13];
ry(-3.0347941299557397) q[14];
rz(-1.2592696889495691) q[14];
ry(2.443022632238776) q[15];
rz(-3.089286318212299) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.5411681282355228) q[0];
rz(0.5793454448417779) q[0];
ry(-1.1180141526481389) q[1];
rz(1.0676617130861936) q[1];
ry(2.0033868880065446) q[2];
rz(-2.0060578427167375) q[2];
ry(2.4268141016844624) q[3];
rz(2.902644238871243) q[3];
ry(1.814666316037558) q[4];
rz(0.09129738422957878) q[4];
ry(1.2814683525730848) q[5];
rz(-0.25754353506838307) q[5];
ry(-1.0672416486552188) q[6];
rz(1.1313983854131633) q[6];
ry(0.712816056239418) q[7];
rz(-0.09273347247840213) q[7];
ry(0.6121872482846492) q[8];
rz(-0.9405016367550805) q[8];
ry(0.3090523146260634) q[9];
rz(0.06350278424457247) q[9];
ry(-0.6010483368674355) q[10];
rz(-0.3041900354366139) q[10];
ry(0.8758135714644757) q[11];
rz(-1.9252403653254557) q[11];
ry(0.26036841627212876) q[12];
rz(1.1661497249840327) q[12];
ry(-2.3069770489129855) q[13];
rz(0.10236974461826966) q[13];
ry(-2.6039730614917866) q[14];
rz(-1.7362230384089672) q[14];
ry(2.921440387109356) q[15];
rz(-2.777860142479776) q[15];