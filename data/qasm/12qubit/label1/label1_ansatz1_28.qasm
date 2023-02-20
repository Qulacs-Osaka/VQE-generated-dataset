OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.08824901202615722) q[0];
rz(-2.530692508334691) q[0];
ry(1.4600936059518181) q[1];
rz(1.8707865880860781) q[1];
ry(-2.35669725915267) q[2];
rz(0.11892323737777577) q[2];
ry(-2.615570896954224) q[3];
rz(-1.6463403424206964) q[3];
ry(2.842943128006941) q[4];
rz(1.684397776206339) q[4];
ry(0.7148310439154061) q[5];
rz(2.0859703827617233) q[5];
ry(-2.190387275776108) q[6];
rz(-1.611937681681079) q[6];
ry(-2.441785439248189) q[7];
rz(-2.4798451014808807) q[7];
ry(2.737423922405339) q[8];
rz(-0.7430828431179455) q[8];
ry(0.028417107938095043) q[9];
rz(2.062385414475903) q[9];
ry(2.7406374592282288) q[10];
rz(-1.1029474777067865) q[10];
ry(-0.9223367868537276) q[11];
rz(-2.415192827148825) q[11];
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
ry(-1.4232256862282555) q[0];
rz(2.6150799727774214) q[0];
ry(-0.9375200046071156) q[1];
rz(-0.30574841668847397) q[1];
ry(-3.062002053268203) q[2];
rz(-0.41825091219661026) q[2];
ry(2.9931320100181207) q[3];
rz(-3.12350102678983) q[3];
ry(2.9151783074096063) q[4];
rz(-0.19802709172969496) q[4];
ry(-2.137218130395307) q[5];
rz(0.7444418338167885) q[5];
ry(-2.6766823410750513) q[6];
rz(-1.2001477201212425) q[6];
ry(2.739204390739039) q[7];
rz(-2.972944589052922) q[7];
ry(2.0184548268604203) q[8];
rz(2.9914317555371435) q[8];
ry(-0.009491692157483755) q[9];
rz(0.7570687976985756) q[9];
ry(-0.5083355494241966) q[10];
rz(1.212413056345311) q[10];
ry(1.9828651344976356) q[11];
rz(0.9069575188350366) q[11];
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
ry(0.6250383845545269) q[0];
rz(1.8290286458907044) q[0];
ry(2.75191492476481) q[1];
rz(-1.7900032777495296) q[1];
ry(1.615326946143794) q[2];
rz(0.4605726245575377) q[2];
ry(-1.5457306772816757) q[3];
rz(0.5643050222840849) q[3];
ry(-0.423665922128662) q[4];
rz(-2.7198671396671545) q[4];
ry(-2.3760647972913067) q[5];
rz(1.600463409931504) q[5];
ry(1.0745162329144824) q[6];
rz(1.4269873075314803) q[6];
ry(2.901522838057476) q[7];
rz(-1.4313165041420253) q[7];
ry(-2.8385829782048817) q[8];
rz(-1.4351483032810652) q[8];
ry(0.002597975093292959) q[9];
rz(2.0809835253828917) q[9];
ry(1.4704572786194199) q[10];
rz(-0.5266158198714008) q[10];
ry(-1.3548736165322446) q[11];
rz(-0.3587395022337363) q[11];
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
ry(-1.411361062534824) q[0];
rz(0.20791319173057238) q[0];
ry(3.1323768090448763) q[1];
rz(1.9186153753509332) q[1];
ry(-3.1298493569219725) q[2];
rz(-1.2656141206247735) q[2];
ry(0.08306181752752995) q[3];
rz(2.5937924573599034) q[3];
ry(-0.09710611660582646) q[4];
rz(-0.9703329700278743) q[4];
ry(0.2833431691120146) q[5];
rz(-2.9529717904503063) q[5];
ry(2.1466612148523874) q[6];
rz(1.5267323075324382) q[6];
ry(2.194051826532042) q[7];
rz(-2.8876586754900364) q[7];
ry(-0.186222611990664) q[8];
rz(-1.6743831556339868) q[8];
ry(-3.133555533473135) q[9];
rz(1.3540517595156902) q[9];
ry(-2.3479600273448717) q[10];
rz(-2.1879655107527953) q[10];
ry(-0.45286577227520297) q[11];
rz(-1.5298675180975385) q[11];
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
ry(-1.5009695588385163) q[0];
rz(-1.64932002696482) q[0];
ry(-3.045858895472831) q[1];
rz(1.718236674529299) q[1];
ry(-1.03650182997663) q[2];
rz(-0.03731187666463197) q[2];
ry(1.8327294777017753) q[3];
rz(-2.3428623664117105) q[3];
ry(2.8964605219699586) q[4];
rz(-2.4309354479445773) q[4];
ry(-2.073240686388378) q[5];
rz(-0.9772305694556824) q[5];
ry(1.7263416968292693) q[6];
rz(-2.988782823004652) q[6];
ry(-0.022830764853337016) q[7];
rz(0.9948082413660986) q[7];
ry(-0.19227454669579025) q[8];
rz(2.3637272055552816) q[8];
ry(-0.007300136301573129) q[9];
rz(-2.8765597822496196) q[9];
ry(-0.9199024199016899) q[10];
rz(-1.444390038634375) q[10];
ry(-1.8551981834488371) q[11];
rz(2.9093984747031785) q[11];
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
ry(2.2165425273651707) q[0];
rz(2.9998479321086715) q[0];
ry(-2.190318271614345) q[1];
rz(-1.655389330221734) q[1];
ry(-3.1146010142942893) q[2];
rz(-2.4251581037960657) q[2];
ry(3.119204676412942) q[3];
rz(0.1697264996671954) q[3];
ry(-2.9014321507355683) q[4];
rz(-2.842761644482994) q[4];
ry(1.1727459094070625) q[5];
rz(-0.04878249603992781) q[5];
ry(-1.173027786111077) q[6];
rz(0.32968641536974136) q[6];
ry(1.835694333784904) q[7];
rz(1.772658426700655) q[7];
ry(2.113208206637541) q[8];
rz(-2.403312162488094) q[8];
ry(0.013281233948515908) q[9];
rz(-0.19965742196269698) q[9];
ry(-0.3161970816036214) q[10];
rz(-0.7893802021183927) q[10];
ry(-0.5580163149416775) q[11];
rz(-2.4827884450892634) q[11];
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
ry(-3.104179855017311) q[0];
rz(1.845309138225305) q[0];
ry(-1.7475619723980023) q[1];
rz(1.9327046423860441) q[1];
ry(2.571235240545607) q[2];
rz(0.5992832747707779) q[2];
ry(-0.7978792533378435) q[3];
rz(-0.7182538161160164) q[3];
ry(1.0171178196195303) q[4];
rz(3.034140006401294) q[4];
ry(1.6077624958506584) q[5];
rz(3.0688759500596183) q[5];
ry(2.2240260741008058) q[6];
rz(-1.1733770108421133) q[6];
ry(-0.3092507095693975) q[7];
rz(-0.9826967229125734) q[7];
ry(0.4920295287733922) q[8];
rz(1.1719794690539156) q[8];
ry(0.13942598808060502) q[9];
rz(2.2287472474081) q[9];
ry(-0.5986464281185842) q[10];
rz(2.818908209940902) q[10];
ry(-2.778400001723465) q[11];
rz(0.5195934014503294) q[11];
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
ry(-0.8233230544666352) q[0];
rz(-2.819315508770826) q[0];
ry(-3.134903510083663) q[1];
rz(-2.5022590986615136) q[1];
ry(-2.3086378664193408) q[2];
rz(1.3068945122939328) q[2];
ry(-3.1294743417044737) q[3];
rz(-3.0904072032713334) q[3];
ry(-3.0674842104988302) q[4];
rz(2.629287870414752) q[4];
ry(1.9150496732657363) q[5];
rz(2.56807222148987) q[5];
ry(1.2359587650670036) q[6];
rz(1.5058057342586142) q[6];
ry(0.8186585157359383) q[7];
rz(2.9427092101841272) q[7];
ry(-1.4492502980305835) q[8];
rz(-1.7288891758807712) q[8];
ry(-0.00015514327154842605) q[9];
rz(1.6403318750075613) q[9];
ry(3.04751170313277) q[10];
rz(-2.956458747526879) q[10];
ry(-0.13408274851186786) q[11];
rz(2.3138516969912466) q[11];
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
ry(-1.3995861839345496) q[0];
rz(1.5488867028387217) q[0];
ry(0.8715790384988356) q[1];
rz(3.0244834607095528) q[1];
ry(2.9105389527583863) q[2];
rz(-2.0082021647188175) q[2];
ry(-3.0895573813712374) q[3];
rz(1.3556614171239616) q[3];
ry(1.660801433281013) q[4];
rz(-2.523930813053914) q[4];
ry(-2.1158268137560894) q[5];
rz(-0.7121434370000346) q[5];
ry(-2.6708147996241434) q[6];
rz(-0.5375077859913899) q[6];
ry(0.17281987970091794) q[7];
rz(-2.045614661536897) q[7];
ry(0.07865896560938697) q[8];
rz(-0.8519503956066786) q[8];
ry(0.08052436793046705) q[9];
rz(2.196790200809231) q[9];
ry(-0.7846317604818779) q[10];
rz(-2.687522139757297) q[10];
ry(1.6754497643190394) q[11];
rz(-2.2950149209428345) q[11];
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
ry(2.698309949617621) q[0];
rz(-2.037362032633487) q[0];
ry(0.061019945841955796) q[1];
rz(-3.019122416072632) q[1];
ry(-0.1281003838546484) q[2];
rz(1.8588728345966903) q[2];
ry(3.0951287106201404) q[3];
rz(-0.2676546768397516) q[3];
ry(-0.10316841004770956) q[4];
rz(-0.7919623656479864) q[4];
ry(-2.1134130813658736) q[5];
rz(-1.317570875205409) q[5];
ry(2.6550315383022123) q[6];
rz(-0.97032262351211) q[6];
ry(1.3606968593614084) q[7];
rz(0.3880815214739642) q[7];
ry(0.4283646865592967) q[8];
rz(-0.06612227013975004) q[8];
ry(-2.355210911125849) q[9];
rz(-3.117623032755472) q[9];
ry(-2.7075313226710027) q[10];
rz(2.1921708817250787) q[10];
ry(0.19358807622268426) q[11];
rz(-0.1351087071626225) q[11];
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
ry(-1.7605085256490118) q[0];
rz(-2.132253950985942) q[0];
ry(1.2109003741464877) q[1];
rz(0.7283222595875724) q[1];
ry(-2.387856355551626) q[2];
rz(2.94833820506025) q[2];
ry(-2.0654805303625463) q[3];
rz(-3.0529894406441427) q[3];
ry(1.4181539953693612) q[4];
rz(-0.7544347573757952) q[4];
ry(-0.829602460851099) q[5];
rz(1.3903820763170274) q[5];
ry(1.7050466457570241) q[6];
rz(-2.8554169809230694) q[6];
ry(3.0298268169211404) q[7];
rz(-1.893071931339967) q[7];
ry(3.0905024582612914) q[8];
rz(-2.4323825847985527) q[8];
ry(-0.0751878209513275) q[9];
rz(0.43300952129241255) q[9];
ry(2.845341392584241) q[10];
rz(0.09324946108275077) q[10];
ry(0.32079215957951135) q[11];
rz(0.14117050860023106) q[11];
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
ry(1.269087813370719) q[0];
rz(-2.934330170970479) q[0];
ry(0.4431648072322725) q[1];
rz(0.9115680219862359) q[1];
ry(0.01307380259480426) q[2];
rz(3.1270611729205386) q[2];
ry(0.3966309551178071) q[3];
rz(-1.1628751678668303) q[3];
ry(1.860244469068241) q[4];
rz(0.4356994795487905) q[4];
ry(-0.015204873468578347) q[5];
rz(2.9520006880874776) q[5];
ry(1.0730970439825618) q[6];
rz(2.826428496457326) q[6];
ry(-2.6899963189336447) q[7];
rz(1.5870096466848798) q[7];
ry(2.158045763400595) q[8];
rz(2.043855782306693) q[8];
ry(0.17795339467670515) q[9];
rz(-1.499817848489502) q[9];
ry(1.841174691793396) q[10];
rz(3.1361875751315886) q[10];
ry(1.9283520846987112) q[11];
rz(-1.7105376054954151) q[11];
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
ry(-0.6375364236981538) q[0];
rz(1.9140994300985765) q[0];
ry(3.1261990915829254) q[1];
rz(-0.2661432486341709) q[1];
ry(1.5878925903263275) q[2];
rz(1.241800725321594) q[2];
ry(-1.3057935133852405) q[3];
rz(-2.13329712646795) q[3];
ry(-2.971932306385996) q[4];
rz(-2.589061801169664) q[4];
ry(0.013372064402141799) q[5];
rz(-0.057653019315427316) q[5];
ry(1.1342539189117762) q[6];
rz(-2.1723805107930954) q[6];
ry(2.0721879735645095) q[7];
rz(-1.9756584019248469) q[7];
ry(0.08240892505812406) q[8];
rz(0.7529656595829577) q[8];
ry(2.893690891875377) q[9];
rz(1.7687670311505272) q[9];
ry(-2.89997943466165) q[10];
rz(-2.1242929359449336) q[10];
ry(-1.8892521672742626) q[11];
rz(1.3204119615869212) q[11];
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
ry(-1.4893867531315945) q[0];
rz(0.140365310268618) q[0];
ry(-3.1345557608458425) q[1];
rz(0.9972251827869361) q[1];
ry(2.920052758968029) q[2];
rz(-2.5595083965107435) q[2];
ry(3.0252384492344513) q[3];
rz(-0.6533472552085121) q[3];
ry(0.8427204572653764) q[4];
rz(-2.6184640426130192) q[4];
ry(-0.24128125495825853) q[5];
rz(-1.4892707188403707) q[5];
ry(1.3291455171273647) q[6];
rz(-0.15161096257899584) q[6];
ry(-0.8104931732271522) q[7];
rz(0.7670894109936598) q[7];
ry(-0.1628422266101932) q[8];
rz(-0.549525142428) q[8];
ry(0.6137843806569219) q[9];
rz(-1.5033303569654997) q[9];
ry(-0.48173112857605105) q[10];
rz(-2.749085152858772) q[10];
ry(-1.7424797805558452) q[11];
rz(-0.6254376275063347) q[11];
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
ry(-2.5202328919681882) q[0];
rz(-0.6305105766705049) q[0];
ry(-0.25895811507059285) q[1];
rz(2.8611584362269604) q[1];
ry(1.4179668220085997) q[2];
rz(-3.1376928070425496) q[2];
ry(-0.35174689562034417) q[3];
rz(-1.8733812052958627) q[3];
ry(1.3011469674589438) q[4];
rz(0.0957177826600697) q[4];
ry(-3.127147687109575) q[5];
rz(2.8684496149306002) q[5];
ry(-2.9320848404971946) q[6];
rz(2.7532359418240504) q[6];
ry(-2.1255490839405153) q[7];
rz(0.7575327360862106) q[7];
ry(2.9972836287449054) q[8];
rz(2.3096354200386573) q[8];
ry(1.9774131003222923) q[9];
rz(-2.695531566934746) q[9];
ry(0.8649802180141846) q[10];
rz(2.343259276406594) q[10];
ry(-0.45559967279973956) q[11];
rz(-0.09835116055349244) q[11];
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
ry(1.6296289950495693) q[0];
rz(-1.5872739316095332) q[0];
ry(-2.8204747627128115) q[1];
rz(-0.35141331292726136) q[1];
ry(-3.0194255699886012) q[2];
rz(-1.7849490115864906) q[2];
ry(3.1297803480008617) q[3];
rz(-1.8579041484063197) q[3];
ry(1.6459099548373006) q[4];
rz(0.0055783281578422605) q[4];
ry(-0.04554217780288659) q[5];
rz(-0.1564896937837341) q[5];
ry(-2.801844360359803) q[6];
rz(0.8091878456818804) q[6];
ry(-1.8637037424449918) q[7];
rz(-2.4714562807435185) q[7];
ry(2.1146403804928955) q[8];
rz(-1.0987759753222426) q[8];
ry(-0.06963961120510989) q[9];
rz(-3.108329232249428) q[9];
ry(0.24216443627417306) q[10];
rz(-1.1669449939921215) q[10];
ry(-3.0155956780878492) q[11];
rz(2.7984062787663206) q[11];
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
ry(0.5151234864838867) q[0];
rz(-2.934397705160646) q[0];
ry(0.017043297216534192) q[1];
rz(2.6055211229579855) q[1];
ry(2.061497303855606) q[2];
rz(-1.849374958413317) q[2];
ry(-1.8963024877353671) q[3];
rz(-1.811754136953745) q[3];
ry(1.2527046037432106) q[4];
rz(0.5595593607675997) q[4];
ry(-0.4701548742788755) q[5];
rz(1.2457458168393414) q[5];
ry(2.9050738653075525) q[6];
rz(1.7564400685510415) q[6];
ry(-3.139378278509064) q[7];
rz(1.3387213225512842) q[7];
ry(-0.979734861653708) q[8];
rz(0.5695290858743809) q[8];
ry(-0.3450954966166391) q[9];
rz(2.5103404669625706) q[9];
ry(-1.8268939352232973) q[10];
rz(2.1018662845472176) q[10];
ry(2.7464178605790095) q[11];
rz(-2.3542515445321044) q[11];
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
ry(-2.682000796201794) q[0];
rz(-2.786885633398057) q[0];
ry(-2.521042960881939) q[1];
rz(2.7299582861566263) q[1];
ry(-0.041894449609840075) q[2];
rz(-0.691389609536236) q[2];
ry(1.0775201340817144) q[3];
rz(-2.694552049483778) q[3];
ry(0.12083768469740441) q[4];
rz(3.0832799872071477) q[4];
ry(-0.2939846002310036) q[5];
rz(-1.561448822421101) q[5];
ry(0.029059585558853133) q[6];
rz(2.0411482367724485) q[6];
ry(1.7479108515128423) q[7];
rz(-0.07214532689786601) q[7];
ry(1.1759319966204451) q[8];
rz(-3.0660934702557725) q[8];
ry(-3.0865479604079002) q[9];
rz(1.219605147815991) q[9];
ry(-0.6108179135727859) q[10];
rz(-3.038427046518251) q[10];
ry(0.37631201365547007) q[11];
rz(-2.8147487049307505) q[11];
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
ry(-0.43896972817874286) q[0];
rz(1.517624553938786) q[0];
ry(3.1323180697556396) q[1];
rz(1.1777776473196626) q[1];
ry(-2.2967662545833245) q[2];
rz(1.6889742225931412) q[2];
ry(-2.2767691621187716) q[3];
rz(-2.4681761807805835) q[3];
ry(-3.105098953997109) q[4];
rz(-1.7562193029513757) q[4];
ry(0.8533814609207652) q[5];
rz(2.6397086915126717) q[5];
ry(3.139796332539568) q[6];
rz(-1.9742317838643197) q[6];
ry(3.135182176147076) q[7];
rz(-2.2399249829393497) q[7];
ry(1.3777121946190363) q[8];
rz(0.7008587213530646) q[8];
ry(1.409459336091615) q[9];
rz(-2.9875095338431596) q[9];
ry(-2.6985930462255716) q[10];
rz(-0.5080201415261263) q[10];
ry(2.9598302832249166) q[11];
rz(0.2836812077014166) q[11];
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
ry(1.8668603660282521) q[0];
rz(0.5300428081765355) q[0];
ry(-0.7558486895251084) q[1];
rz(0.9742085547347639) q[1];
ry(-2.6968851354490733) q[2];
rz(-0.36049384302965926) q[2];
ry(2.5263738128507813) q[3];
rz(1.8312719620596207) q[3];
ry(2.8983764696398375) q[4];
rz(0.7093384135132207) q[4];
ry(-2.940955275045566) q[5];
rz(0.31879427200383736) q[5];
ry(-0.1823505072840126) q[6];
rz(-0.1299888253516463) q[6];
ry(-1.458351708576922) q[7];
rz(-1.1620683781593335) q[7];
ry(-2.7522487080446365) q[8];
rz(2.5618110600540254) q[8];
ry(-0.554671772846139) q[9];
rz(0.369478705190982) q[9];
ry(-2.7766609011523116) q[10];
rz(0.36606847303393947) q[10];
ry(-0.4189243999006414) q[11];
rz(-2.231775814751482) q[11];
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
ry(3.1328299035171487) q[0];
rz(-1.6605530089587235) q[0];
ry(3.0456024106441797) q[1];
rz(-2.4794277256368367) q[1];
ry(-1.5602556087189559) q[2];
rz(0.9463745831292858) q[2];
ry(-3.0300023513389474) q[3];
rz(-0.713799391257507) q[3];
ry(-0.030776773849149563) q[4];
rz(2.354651193333842) q[4];
ry(1.1815589917766518) q[5];
rz(-0.9617426831033966) q[5];
ry(-0.002953813039155498) q[6];
rz(2.8153171430130155) q[6];
ry(0.07065739675099625) q[7];
rz(0.023832006799831523) q[7];
ry(-2.9296250205332415) q[8];
rz(1.0121048681216305) q[8];
ry(2.152416979567114) q[9];
rz(-0.04234236700344951) q[9];
ry(-0.19537245264294956) q[10];
rz(2.634340415806941) q[10];
ry(2.015801149107762) q[11];
rz(-2.903085009997182) q[11];
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
ry(0.7932703865580448) q[0];
rz(3.094562655951138) q[0];
ry(-0.5185276784220078) q[1];
rz(-2.8241287612076555) q[1];
ry(-0.6810196335251508) q[2];
rz(2.37724956101411) q[2];
ry(2.831228134752245) q[3];
rz(1.363583693107195) q[3];
ry(0.16154058241196684) q[4];
rz(0.5508534525462863) q[4];
ry(-3.0324340849006313) q[5];
rz(2.3642893506873133) q[5];
ry(-3.0995459977219135) q[6];
rz(0.9160024323119279) q[6];
ry(-1.8503817135868381) q[7];
rz(-2.75021693942864) q[7];
ry(2.5069822290767236) q[8];
rz(1.8828284913758764) q[8];
ry(2.6077395763021483) q[9];
rz(-1.7715955289907512) q[9];
ry(-3.0731999327369035) q[10];
rz(-2.0983408549685256) q[10];
ry(-0.008344682485442512) q[11];
rz(2.9609928739607128) q[11];
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
ry(0.12626429789470883) q[0];
rz(2.191027041618998) q[0];
ry(0.02579725715550326) q[1];
rz(0.05006830190212241) q[1];
ry(1.1349512572882) q[2];
rz(-1.8195024564806346) q[2];
ry(0.03517617559184938) q[3];
rz(1.118116209916672) q[3];
ry(-2.2150846315953894) q[4];
rz(1.197536042235341) q[4];
ry(-1.7211952310248937) q[5];
rz(0.13148415962319568) q[5];
ry(3.090369205372573) q[6];
rz(-1.7919451240001232) q[6];
ry(0.05074088623778561) q[7];
rz(0.01586879425051148) q[7];
ry(0.8934615563275248) q[8];
rz(0.9683311214861164) q[8];
ry(1.50016714226736) q[9];
rz(-1.6007553687637524) q[9];
ry(-2.0108449998493994) q[10];
rz(-0.7865207975189339) q[10];
ry(-1.2430629506190274) q[11];
rz(1.640038449576946) q[11];
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
ry(-2.1573632096245623) q[0];
rz(-1.215124125369238) q[0];
ry(-0.7092048009847722) q[1];
rz(-0.9686396334845409) q[1];
ry(-0.028853807729937862) q[2];
rz(1.5220951568849719) q[2];
ry(-2.895426057907453) q[3];
rz(1.680886390192498) q[3];
ry(-0.04011120177989902) q[4];
rz(1.381439253283987) q[4];
ry(-2.4790395673630576) q[5];
rz(-0.1271870348353854) q[5];
ry(2.1543123970488782) q[6];
rz(-0.19159586052693456) q[6];
ry(-2.988697823936394) q[7];
rz(-2.087706043670009) q[7];
ry(2.127502609856193) q[8];
rz(0.9245042387483493) q[8];
ry(-0.46632762691586666) q[9];
rz(0.4960457002709342) q[9];
ry(-1.152928130542351) q[10];
rz(0.31461762180350217) q[10];
ry(-3.0811962513321944) q[11];
rz(-2.2256184037695483) q[11];
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
ry(2.779591122967911) q[0];
rz(-0.5636466419154861) q[0];
ry(1.8915975421369164) q[1];
rz(-2.320888148469752) q[1];
ry(-0.12273323026516046) q[2];
rz(-0.024598824073812517) q[2];
ry(-3.119716903057925) q[3];
rz(-1.524606983917419) q[3];
ry(-2.980852176705738) q[4];
rz(-1.7697542730481217) q[4];
ry(-2.370096948654214) q[5];
rz(-0.056852048531348336) q[5];
ry(-3.1228668149716308) q[6];
rz(-2.586520849596986) q[6];
ry(3.1204922920227136) q[7];
rz(-1.1356494778503403) q[7];
ry(1.2283369469287484) q[8];
rz(-0.042095341883530146) q[8];
ry(1.3302792230919112) q[9];
rz(-1.5733931578734213) q[9];
ry(-1.6512299912310089) q[10];
rz(-0.7944486999558645) q[10];
ry(-1.0057888013913532) q[11];
rz(0.2751115894816918) q[11];
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
ry(1.1634575169977337) q[0];
rz(0.8971349161162917) q[0];
ry(3.049385314506768) q[1];
rz(0.7954156517009565) q[1];
ry(3.139309374917634) q[2];
rz(1.446744543636454) q[2];
ry(-0.9961156567573539) q[3];
rz(0.3634090524737426) q[3];
ry(3.108097018968948) q[4];
rz(-1.1893516331236684) q[4];
ry(2.4374874892551612) q[5];
rz(3.016375533937005) q[5];
ry(1.7226592721688017) q[6];
rz(-1.2733710606456425) q[6];
ry(-0.12265101161054126) q[7];
rz(2.1865165874469286) q[7];
ry(2.8579399873598472) q[8];
rz(0.04391196043009238) q[8];
ry(0.5590222573989517) q[9];
rz(0.6909549945437918) q[9];
ry(0.11603640499659644) q[10];
rz(0.33783809381094804) q[10];
ry(-2.0985238011982252) q[11];
rz(-2.6136394507761005) q[11];
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
ry(-0.49299980491731604) q[0];
rz(-1.6595458603310878) q[0];
ry(-1.8722736012628718) q[1];
rz(2.1651137677807233) q[1];
ry(0.012540533538294889) q[2];
rz(-0.6157262860396973) q[2];
ry(-3.068157556874186) q[3];
rz(-2.715845142119064) q[3];
ry(-1.1067368303187077) q[4];
rz(-3.0248750506497117) q[4];
ry(1.2521816032065463) q[5];
rz(-0.15713882801408555) q[5];
ry(0.026420545357835223) q[6];
rz(-1.8973813929445387) q[6];
ry(-0.010704778363233025) q[7];
rz(0.3462577155500516) q[7];
ry(-2.1531887741875826) q[8];
rz(-1.8354167313957985) q[8];
ry(-0.6967609320635786) q[9];
rz(-2.2418773141012203) q[9];
ry(-2.9176525585457105) q[10];
rz(-1.942710689543013) q[10];
ry(-1.1337220876856087) q[11];
rz(2.8921505430540533) q[11];
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
ry(-0.48422742159133225) q[0];
rz(-2.2502536360919585) q[0];
ry(1.6363111408801005) q[1];
rz(-2.0177012452322645) q[1];
ry(-1.3320950518906733) q[2];
rz(0.24457543174401067) q[2];
ry(1.0277127304003173) q[3];
rz(-1.4906186132337236) q[3];
ry(-3.124066334622756) q[4];
rz(-3.0930767554026137) q[4];
ry(-3.1190014361987517) q[5];
rz(-0.2733704493177118) q[5];
ry(0.9972957786783508) q[6];
rz(1.9632634253306414) q[6];
ry(-2.300445960753683) q[7];
rz(1.364655823974857) q[7];
ry(-0.9560727095083017) q[8];
rz(-3.135283143901756) q[8];
ry(1.8503126452068626) q[9];
rz(0.8519243944625217) q[9];
ry(-1.1493491211034004) q[10];
rz(2.4133912143389726) q[10];
ry(1.0398732038504381) q[11];
rz(-2.459900440962897) q[11];
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
ry(-2.5649736962422094) q[0];
rz(-3.0871453087252667) q[0];
ry(-0.009192795038691415) q[1];
rz(1.1901320441185572) q[1];
ry(2.8842870667433247) q[2];
rz(1.4431200474233832) q[2];
ry(-3.0967560775372887) q[3];
rz(-1.3265761440752046) q[3];
ry(-0.6605062842218983) q[4];
rz(-1.2698068815513786) q[4];
ry(-1.9276145934425553) q[5];
rz(-1.2616447165134892) q[5];
ry(-0.10303226927905201) q[6];
rz(1.1694960221790922) q[6];
ry(3.121645249206717) q[7];
rz(1.5703134965292025) q[7];
ry(-2.964502977810228) q[8];
rz(-1.7575265204955883) q[8];
ry(-0.7117784116943777) q[9];
rz(1.9100760412964959) q[9];
ry(1.2558073364759688) q[10];
rz(-1.9305063269199332) q[10];
ry(2.6324989886762333) q[11];
rz(-2.188845691394141) q[11];
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
ry(2.1814431683641993) q[0];
rz(-1.306450776684116) q[0];
ry(1.7504794036505016) q[1];
rz(3.0649991364871645) q[1];
ry(0.5566298380800943) q[2];
rz(-2.401660004387136) q[2];
ry(1.5986123168085422) q[3];
rz(1.535100803903507) q[3];
ry(-1.7558117430410303) q[4];
rz(0.911393458599667) q[4];
ry(-1.8215512546654473) q[5];
rz(-2.598631516246178) q[5];
ry(-0.9091015308471491) q[6];
rz(2.629579518663507) q[6];
ry(0.9039584522642894) q[7];
rz(-1.3097238603950634) q[7];
ry(2.956942792965713) q[8];
rz(-1.3814148008006444) q[8];
ry(-0.9618404944247158) q[9];
rz(-2.55945192690981) q[9];
ry(1.3312515992936929) q[10];
rz(1.93744054475909) q[10];
ry(-3.1146795692179734) q[11];
rz(2.542999896820973) q[11];
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
ry(2.553446617584174) q[0];
rz(0.6905137362265252) q[0];
ry(1.5725534965661179) q[1];
rz(-1.557915492493061) q[1];
ry(0.009448346672583005) q[2];
rz(-0.35166484657072355) q[2];
ry(0.00240588908729723) q[3];
rz(-0.44528260659954483) q[3];
ry(0.03476116538318941) q[4];
rz(0.59712295951784) q[4];
ry(3.1168250111212514) q[5];
rz(2.006488729398433) q[5];
ry(0.018003672291716157) q[6];
rz(0.42116126538974064) q[6];
ry(-0.05798126807950051) q[7];
rz(-3.0078862673816325) q[7];
ry(2.9113114437464636) q[8];
rz(1.3304112498360903) q[8];
ry(-0.08193813522413793) q[9];
rz(0.21742451828749954) q[9];
ry(0.14296288137316307) q[10];
rz(-0.29946058044499857) q[10];
ry(-2.5776203707341905) q[11];
rz(0.5205327448066134) q[11];
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
ry(-0.007796502499290477) q[0];
rz(2.998104973999984) q[0];
ry(-1.568101964579201) q[1];
rz(-2.8568992098007184) q[1];
ry(-1.5604824157539163) q[2];
rz(0.39551917713284684) q[2];
ry(0.023254236793787134) q[3];
rz(0.012275058570599292) q[3];
ry(1.9327383002325238) q[4];
rz(-0.07613331673029487) q[4];
ry(-1.0590235003413238) q[5];
rz(-1.0532701019749222) q[5];
ry(-0.47700265830451505) q[6];
rz(0.4644155793476121) q[6];
ry(1.24556553070311) q[7];
rz(-0.1750069648638266) q[7];
ry(1.9191876082108803) q[8];
rz(2.747155711679982) q[8];
ry(-1.8154123187837468) q[9];
rz(-1.3006451065326665) q[9];
ry(-1.5993663351961844) q[10];
rz(1.1615549400958765) q[10];
ry(3.072883492669663) q[11];
rz(-1.6332864833434928) q[11];