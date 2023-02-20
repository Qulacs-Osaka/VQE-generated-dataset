OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.7854431233018355) q[0];
rz(-0.2004179389296095) q[0];
ry(1.482352560497186) q[1];
rz(1.1594577806355257) q[1];
ry(-1.568274457137176) q[2];
rz(-1.550755272657299) q[2];
ry(0.14801380362411856) q[3];
rz(-0.3979965497959063) q[3];
ry(1.5708285779055435) q[4];
rz(-0.16394510632887638) q[4];
ry(0.0019805303131406543) q[5];
rz(2.8222933435942674) q[5];
ry(1.5708828460712803) q[6];
rz(1.5708070312785953) q[6];
ry(-0.8620030604728042) q[7];
rz(0.3596327001225718) q[7];
ry(1.570824112352734) q[8];
rz(-1.5708836987670343) q[8];
ry(1.9857459648115068) q[9];
rz(-0.5271300947485523) q[9];
ry(-1.5470489398855938) q[10];
rz(-1.6071375705439988) q[10];
ry(-1.3743805180354502) q[11];
rz(-1.0218031121228126) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.3989324044869607) q[0];
rz(0.9088021431652604) q[0];
ry(-0.008381961680473346) q[1];
rz(-0.017831137423482524) q[1];
ry(-1.313565893004346) q[2];
rz(1.736488334240752) q[2];
ry(-1.5708440149657525) q[3];
rz(1.6288020448689355) q[3];
ry(4.4566454453054405e-05) q[4];
rz(-1.4064684303102042) q[4];
ry(-1.5709922026633611) q[5];
rz(3.1204667205367564) q[5];
ry(-2.7002223002542576) q[6];
rz(2.9734273173146203e-05) q[6];
ry(-2.371458007300131) q[7];
rz(-1.9448366720249564) q[7];
ry(-1.0984112318626753) q[8];
rz(2.2365489563469506e-05) q[8];
ry(3.1412947185713174) q[9];
rz(-0.5726324186339397) q[9];
ry(-3.09348601101707) q[10];
rz(3.099134541561566) q[10];
ry(-3.0814329949461747) q[11];
rz(2.8274861768920965) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.201734873730921) q[0];
rz(2.6087679145139737) q[0];
ry(-2.7577727092329765) q[1];
rz(-0.18465875303585375) q[1];
ry(-3.1415539917719113) q[2];
rz(-1.5439609545942785) q[2];
ry(2.4533513718489317) q[3];
rz(0.07541872033831522) q[3];
ry(1.4196625554741527) q[4];
rz(3.1415534400094196) q[4];
ry(-1.7313792050824668) q[5];
rz(0.020956823326423447) q[5];
ry(1.5702434010621622) q[6];
rz(-2.977521722972801) q[6];
ry(-3.035652247957064) q[7];
rz(-1.891753599670461) q[7];
ry(-1.5341731273005625) q[8];
rz(0.00014052973163281024) q[8];
ry(-0.979722911385208) q[9];
rz(-0.173685813279306) q[9];
ry(0.40201235657413825) q[10];
rz(-3.111101575618195) q[10];
ry(-2.1527384073242324) q[11];
rz(-0.03679502994544205) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.3050659480116051) q[0];
rz(-2.865285645579872) q[0];
ry(-1.5898639457481403) q[1];
rz(1.559232961395348) q[1];
ry(8.307445615240994e-05) q[2];
rz(1.7099851246772908) q[2];
ry(0.814466418816991) q[3];
rz(0.0010659013978759294) q[3];
ry(-2.202867191892325) q[4];
rz(3.141518610464774) q[4];
ry(-3.1327824492629834) q[5];
rz(0.02107695293083842) q[5];
ry(0.006439903980304879) q[6];
rz(2.9710215145997814) q[6];
ry(1.5705444357984906) q[7];
rz(3.141431780020886) q[7];
ry(1.4431405671106459) q[8];
rz(-3.1414406556758414) q[8];
ry(-1.5704139395350354) q[9];
rz(-1.5702470649340796) q[9];
ry(-2.965424365759763) q[10];
rz(0.4277092117464648) q[10];
ry(-3.09314213110594) q[11];
rz(-1.4162943764896587) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.0020507942408704853) q[0];
rz(1.3976810143376381) q[0];
ry(1.5672935281865537) q[1];
rz(-1.5597636173964424) q[1];
ry(1.5713868906898734) q[2];
rz(3.1415515892767973) q[2];
ry(-2.0218364981334362) q[3];
rz(-2.9976142202691998) q[3];
ry(-1.7890281397134165) q[4];
rz(0.00011076101364615454) q[4];
ry(1.3697994696647666) q[5];
rz(0.32169581278149917) q[5];
ry(-0.03941909343396849) q[6];
rz(-1.3254134269562612) q[6];
ry(0.36858043820346703) q[7];
rz(0.00018984108359276772) q[7];
ry(-0.7753935568736985) q[8];
rz(3.1415486653012366) q[8];
ry(1.570070990239068) q[9];
rz(0.987816212105765) q[9];
ry(-3.140908712920616) q[10];
rz(-2.721703241018854) q[10];
ry(-1.52099843763247) q[11];
rz(-1.0731570837128004) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.5691698102847493) q[0];
rz(-3.135726147933699) q[0];
ry(-1.5706809857468707) q[1];
rz(-1.5707683182687135) q[1];
ry(-1.5707767542338813) q[2];
rz(-1.570636164181375) q[2];
ry(1.570681200413763) q[3];
rz(-1.5708061470178754) q[3];
ry(-1.689153320890692) q[4];
rz(-1.5708069126035193) q[4];
ry(0.0018664468380835688) q[5];
rz(-1.9389484110004027) q[5];
ry(-0.0003000869029847228) q[6];
rz(-0.23893954604517198) q[6];
ry(-1.5714184302444787) q[7];
rz(-1.570743219898785) q[7];
ry(1.570697317531888) q[8];
rz(-1.5708021263909957) q[8];
ry(1.5706071331966742) q[9];
rz(-1.5726026697460476) q[9];
ry(-1.5708726357941734) q[10];
rz(1.5708778809098514) q[10];
ry(3.1283082605862083) q[11];
rz(-0.8192114902036227) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5708124474677538) q[0];
rz(-3.028420994959996) q[0];
ry(-1.7136301258719389) q[1];
rz(3.0739691771051962) q[1];
ry(-1.5707904474249854) q[2];
rz(-1.457000734397364) q[2];
ry(-1.5707881284021337) q[3];
rz(1.5032150404780475) q[3];
ry(-1.570801220766656) q[4];
rz(-3.027688495825198) q[4];
ry(1.6507315295218108) q[5];
rz(2.5470244085256737) q[5];
ry(-1.5707449110201395) q[6];
rz(-3.048179806068434) q[6];
ry(1.5708256124936215) q[7];
rz(-0.06769375371349007) q[7];
ry(1.5707946979943415) q[8];
rz(1.6640381616126971) q[8];
ry(-1.5707838687207032) q[9];
rz(1.5029363221625547) q[9];
ry(-1.5702152469132118) q[10];
rz(1.6635370068060067) q[10];
ry(1.570831336092361) q[11];
rz(-0.06776770359780811) q[11];