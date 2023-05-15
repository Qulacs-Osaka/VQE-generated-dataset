OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.00718160986539209) q[0];
rz(-1.496612058975875) q[0];
ry(1.586890162017248) q[1];
rz(-1.4883606691584779) q[1];
ry(-3.1138557259553035) q[2];
rz(-0.7120339534019853) q[2];
ry(3.013260100853543) q[3];
rz(-1.2985583187013494) q[3];
ry(1.588631869333501) q[4];
rz(0.5711530388181574) q[4];
ry(-0.23402465674485826) q[5];
rz(-1.6083725855369053) q[5];
ry(-3.084869588882948) q[6];
rz(-0.056578234589232494) q[6];
ry(2.2866259487580822) q[7];
rz(1.595148062729064) q[7];
ry(-0.00019474421177223384) q[8];
rz(-0.900645737604588) q[8];
ry(-0.03192451239479599) q[9];
rz(-1.04766682971176) q[9];
ry(8.706755982447786e-05) q[10];
rz(-1.2774327277039825) q[10];
ry(3.1377529821053973) q[11];
rz(0.12567744261680147) q[11];
ry(0.8572615481357442) q[12];
rz(2.332330245514688) q[12];
ry(-1.990160721104856) q[13];
rz(-2.612464891545041) q[13];
ry(-3.1397856504688533) q[14];
rz(-1.687882432412157) q[14];
ry(2.9358756998343987) q[15];
rz(0.49064729964859455) q[15];
ry(-3.1388433087360883) q[16];
rz(3.059315728303364) q[16];
ry(0.027619549473597346) q[17];
rz(-1.4234451427483004) q[17];
ry(2.937302172446052) q[18];
rz(-1.9142655476502037) q[18];
ry(-1.2190701647849989) q[19];
rz(2.9649229292213555) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.24406379905093803) q[0];
rz(1.1152949881413288) q[0];
ry(3.0273758227554035) q[1];
rz(-1.499640031098914) q[1];
ry(-0.23202720057201096) q[2];
rz(1.1043158781845746) q[2];
ry(0.06706698245167075) q[3];
rz(-2.604574922828355) q[3];
ry(3.1220883745618986) q[4];
rz(1.9365979839431686) q[4];
ry(-0.3070405764257114) q[5];
rz(-1.5946937740592433) q[5];
ry(-1.5720943100793017) q[6];
rz(1.556878314557147) q[6];
ry(-1.444303464770166) q[7];
rz(-1.5847446563827587) q[7];
ry(-2.8921705249653185) q[8];
rz(0.03474137064291659) q[8];
ry(1.561018078314531) q[9];
rz(3.134438069047243) q[9];
ry(0.0002831608742123048) q[10];
rz(-1.6751130210134186) q[10];
ry(-3.1399786367739746) q[11];
rz(-1.3260522696650083) q[11];
ry(0.03484988081900742) q[12];
rz(1.080275940684756) q[12];
ry(2.814601719582467) q[13];
rz(-2.50473074343849) q[13];
ry(-3.1408904533380597) q[14];
rz(-0.3672827734448217) q[14];
ry(0.08642891798143366) q[15];
rz(-2.722080026000484) q[15];
ry(3.139688803363544) q[16];
rz(-1.6138729491661317) q[16];
ry(-0.4377777597585077) q[17];
rz(-0.07707023207389808) q[17];
ry(0.6792740163951595) q[18];
rz(-3.059067409178887) q[18];
ry(-1.1841348811726071) q[19];
rz(-0.6836147029372253) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5843292144081962) q[0];
rz(-1.5747064070721084) q[0];
ry(1.5705147501541026) q[1];
rz(2.5772574780212083) q[1];
ry(-1.608207303334009) q[2];
rz(3.04657123600161) q[2];
ry(1.6811787667373481) q[3];
rz(2.9896488501092664) q[3];
ry(2.965724060281492) q[4];
rz(0.04579672528818968) q[4];
ry(-0.00030815916428128526) q[5];
rz(3.0060918647722916) q[5];
ry(-1.5603294937194565) q[6];
rz(3.1414660363937794) q[6];
ry(1.341768852361482) q[7];
rz(-3.1415855365736847) q[7];
ry(1.5871720366307676) q[8];
rz(1.5710865151583677) q[8];
ry(1.5774653036944064) q[9];
rz(-1.6734602876031746) q[9];
ry(0.24996951531182562) q[10];
rz(0.48925282380230234) q[10];
ry(-1.5554720057199063) q[11];
rz(-2.5303859055924423) q[11];
ry(2.5093632024311283) q[12];
rz(-0.2977078392065487) q[12];
ry(-0.9105386959499858) q[13];
rz(1.84913154840579) q[13];
ry(-0.0014428601457699692) q[14];
rz(-0.46189618811868605) q[14];
ry(0.005245285396606292) q[15];
rz(-1.1705124749810674) q[15];
ry(-1.5022535306559166) q[16];
rz(2.852547171756929) q[16];
ry(-3.0351065706693396) q[17];
rz(3.008665628508677) q[17];
ry(-2.4848640772924164) q[18];
rz(0.1171788384021139) q[18];
ry(-0.011192136084695825) q[19];
rz(-1.4478357589979152) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.2382937938300076) q[0];
rz(-1.5688179300816882) q[0];
ry(-0.0025740646857022116) q[1];
rz(-0.46030823915953056) q[1];
ry(-3.0322318342617782) q[2];
rz(-0.1104821505974094) q[2];
ry(0.1103825960177085) q[3];
rz(0.052491584649564965) q[3];
ry(-0.0010978221497426062) q[4];
rz(1.04671826566382) q[4];
ry(-0.00011628045835686608) q[5];
rz(3.032749688010399) q[5];
ry(-1.5721619657746801) q[6];
rz(-1.5712385756152643) q[6];
ry(1.5616452336032944) q[7];
rz(1.570597389048916) q[7];
ry(1.5709769993719622) q[8];
rz(0.0076309906622684665) q[8];
ry(1.5703839894642586) q[9];
rz(-3.1400346997066837) q[9];
ry(0.21270383912044485) q[10];
rz(2.938890918759546) q[10];
ry(0.024718132089852945) q[11];
rz(0.31031680402629114) q[11];
ry(1.5702384644765859) q[12];
rz(-3.1410232801014977) q[12];
ry(1.5715318598945225) q[13];
rz(-0.000593660834517377) q[13];
ry(1.5590514614864681) q[14];
rz(-1.0010907683367358) q[14];
ry(-0.2409250241598954) q[15];
rz(1.129551187655304) q[15];
ry(-0.010712405883782677) q[16];
rz(1.8718933013276466) q[16];
ry(-0.015143648375233115) q[17];
rz(0.9744907380102958) q[17];
ry(-3.1358160758484845) q[18];
rz(-0.0065639045725767176) q[18];
ry(2.5226091981522814e-05) q[19];
rz(-2.060563001798835) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.580663697314269) q[0];
rz(1.7748271943951648) q[0];
ry(3.117675362698718) q[1];
rz(1.8672947186121365) q[1];
ry(1.4930056139686778) q[2];
rz(2.2551168833629225) q[2];
ry(1.537310718185598) q[3];
rz(0.705425341402222) q[3];
ry(0.0022209246318167304) q[4];
rz(1.9455394652217386) q[4];
ry(3.1260219353322722) q[5];
rz(1.271476764382364) q[5];
ry(1.5707346057117675) q[6];
rz(0.010954118211611965) q[6];
ry(1.5689713814609236) q[7];
rz(-1.5712993382091414) q[7];
ry(1.572210372350734) q[8];
rz(-1.4385385728205058) q[8];
ry(-1.564471499306852) q[9];
rz(-3.030551877618447) q[9];
ry(3.1414864216982097) q[10];
rz(-1.7215758324481438) q[10];
ry(-0.00017187006023533513) q[11];
rz(0.6419639189914577) q[11];
ry(2.3849006561825523) q[12];
rz(1.5699918823694006) q[12];
ry(0.7322423927704884) q[13];
rz(1.5554869749595062) q[13];
ry(0.00022932376334816495) q[14];
rz(-2.1438750590373337) q[14];
ry(3.1413910412117616) q[15];
rz(0.7182512824589821) q[15];
ry(-1.5800397425325643) q[16];
rz(0.9207573148085108) q[16];
ry(-3.1414160221976606) q[17];
rz(-2.2180524444991465) q[17];
ry(-0.09446077978228028) q[18];
rz(0.1798192568421104) q[18];
ry(0.006433560234784252) q[19];
rz(-0.660499520264544) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.5695474520070718) q[0];
rz(3.1410721396682013) q[0];
ry(-1.5646609222220933) q[1];
rz(-3.13109547409209) q[1];
ry(1.5717874457892558) q[2];
rz(0.003030192380759949) q[2];
ry(-1.569391417924941) q[3];
rz(0.003363399059106875) q[3];
ry(-1.575310532650346) q[4];
rz(1.5803674921300441) q[4];
ry(1.3955977650617637) q[5];
rz(0.011036818288514107) q[5];
ry(0.7098912224274719) q[6];
rz(-0.03535519822518449) q[6];
ry(-1.7014537224971704) q[7];
rz(3.1343893935414258) q[7];
ry(-3.137267561989987) q[8];
rz(-1.4390326353094736) q[8];
ry(-3.0978457482387114) q[9];
rz(1.6816107466135723) q[9];
ry(1.5717030851675629) q[10];
rz(0.0032850519395211113) q[10];
ry(1.1174117410154498) q[11];
rz(1.5887139824553937) q[11];
ry(-0.6062822094420683) q[12];
rz(-1.5660181637214317) q[12];
ry(0.3025658889570195) q[13];
rz(-1.5578246799956883) q[13];
ry(-1.1567363360763134) q[14];
rz(1.5570977333049492) q[14];
ry(-1.5710718283099228) q[15];
rz(-0.0030156370066301004) q[15];
ry(0.0023482120793500982) q[16];
rz(0.6514637156751872) q[16];
ry(-1.5707102712619794) q[17];
rz(-0.0006539125236972283) q[17];
ry(-1.570651048438906) q[18];
rz(-3.1412345939461335) q[18];
ry(-1.5706651688526936) q[19];
rz(-1.5121116656838426e-05) q[19];