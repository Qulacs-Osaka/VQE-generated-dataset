OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.7540621548827597) q[0];
rz(1.6913217797924904) q[0];
ry(1.4490628819123819) q[1];
rz(-1.0317447143371465) q[1];
ry(2.178150379149206) q[2];
rz(2.3244923582539228) q[2];
ry(-0.5840068389749353) q[3];
rz(1.5616697099449484) q[3];
ry(1.4981585679503038) q[4];
rz(3.117890517268535) q[4];
ry(0.03755367883861549) q[5];
rz(1.990483943839049) q[5];
ry(-3.051346463552365) q[6];
rz(1.7264000550766339) q[6];
ry(-0.4321737404767644) q[7];
rz(-3.0154622402259545) q[7];
ry(-3.1415177413190474) q[8];
rz(2.213514806216997) q[8];
ry(3.131861406058516) q[9];
rz(1.881237294632338) q[9];
ry(2.998725461632689) q[10];
rz(2.9098193481711028) q[10];
ry(0.8970888883745403) q[11];
rz(-2.3154476407639053) q[11];
ry(-2.79538403336771) q[12];
rz(-2.8337309144984957) q[12];
ry(3.1404244124405345) q[13];
rz(-0.3683733858332883) q[13];
ry(0.0005214420801591354) q[14];
rz(2.8223370753655512) q[14];
ry(-2.321054066273752) q[15];
rz(1.8605449420031706) q[15];
ry(2.177085488359256) q[16];
rz(-0.7643987597325905) q[16];
ry(2.9146287368612813) q[17];
rz(0.9721533574397939) q[17];
ry(-1.6817209941983675) q[18];
rz(2.462504307395047) q[18];
ry(-0.36834998033390526) q[19];
rz(-2.795384818154327) q[19];
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
ry(-2.2197799596205523) q[0];
rz(-1.3122554780889364) q[0];
ry(-1.1185147756013676) q[1];
rz(-0.7223721917246433) q[1];
ry(3.0924184241896566) q[2];
rz(-1.4910662277925009) q[2];
ry(0.5816090527041139) q[3];
rz(2.6997664239919077) q[3];
ry(-0.37419531254861754) q[4];
rz(-3.1179012066805196) q[4];
ry(1.5562954364992063) q[5];
rz(-1.0941561407146134) q[5];
ry(1.5717688418685745) q[6];
rz(0.7035349205323466) q[6];
ry(-2.9760825745592507) q[7];
rz(1.6813327573990504) q[7];
ry(-3.141057500529434) q[8];
rz(-0.29559079573011904) q[8];
ry(1.5727011709396372) q[9];
rz(-1.575451560429705) q[9];
ry(2.2785350506064397) q[10];
rz(-0.22013793942313822) q[10];
ry(0.8694672599073046) q[11];
rz(1.468592733905786) q[11];
ry(1.9473729901459416) q[12];
rz(-1.8595635896449414) q[12];
ry(-9.976975503445118e-05) q[13];
rz(-0.807574717560411) q[13];
ry(-0.0008953367860611294) q[14];
rz(0.6360853754743981) q[14];
ry(2.3190974777032447) q[15];
rz(0.28739547414923433) q[15];
ry(0.11062872059976404) q[16];
rz(1.9271675455013602) q[16];
ry(0.5553389427880341) q[17];
rz(3.0379734546951713) q[17];
ry(2.8193238665608646) q[18];
rz(-2.7909325549679815) q[18];
ry(2.818660060966982) q[19];
rz(3.1172373326917304) q[19];
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
ry(1.0244679965751757) q[0];
rz(-2.274556766766588) q[0];
ry(0.7035481634539553) q[1];
rz(0.6985072577515347) q[1];
ry(-2.8289494759944374) q[2];
rz(-0.4040899884389432) q[2];
ry(3.054763786842122) q[3];
rz(1.131644053185662) q[3];
ry(-1.5862145209356227) q[4];
rz(-3.139299405829052) q[4];
ry(3.1279487531349437) q[5];
rz(2.239070254603851) q[5];
ry(0.016567812465050527) q[6];
rz(2.4432952551925453) q[6];
ry(1.573768742472165) q[7];
rz(1.61016352274203) q[7];
ry(0.14614512398789292) q[8];
rz(1.7512031259500052) q[8];
ry(-1.5908352478542795) q[9];
rz(-1.5717333467780819) q[9];
ry(-1.595925115497686) q[10];
rz(-1.6024566293051679) q[10];
ry(0.0009190451249487154) q[11];
rz(-2.1669542105599255) q[11];
ry(1.8037891172113643) q[12];
rz(3.0042734195002536) q[12];
ry(-0.004476876976972299) q[13];
rz(0.32487347273592837) q[13];
ry(3.1161028330650593) q[14];
rz(0.985978299408848) q[14];
ry(2.3087768487187326) q[15];
rz(1.9045261738777914) q[15];
ry(1.7622871060795178) q[16];
rz(-1.534167649774032) q[16];
ry(3.027025891735698) q[17];
rz(2.3986856656233435) q[17];
ry(-0.8200731459520432) q[18];
rz(2.095580568648509) q[18];
ry(2.2297999913054634) q[19];
rz(-0.3289119800624418) q[19];
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
ry(-2.264959514258091) q[0];
rz(-2.585222946590974) q[0];
ry(-2.950335269363053) q[1];
rz(-1.9492122921090198) q[1];
ry(-3.1408633995855055) q[2];
rz(1.7543558800605918) q[2];
ry(1.2207203816156804) q[3];
rz(1.221438429692263) q[3];
ry(1.1577435928842796) q[4];
rz(0.0013564480515739774) q[4];
ry(-0.003148448878523169) q[5];
rz(1.3091023352408158) q[5];
ry(-1.5763536657334027) q[6];
rz(-2.628031811758831) q[6];
ry(-2.6677562284922036) q[7];
rz(-0.0008854809970628708) q[7];
ry(2.885607177879371) q[8];
rz(0.18448608979459455) q[8];
ry(-1.5677390296952165) q[9];
rz(0.5082475207642952) q[9];
ry(3.139226777342449) q[10];
rz(1.7890990333194212) q[10];
ry(0.6155641837250228) q[11];
rz(1.9821887840835526) q[11];
ry(-2.3533282240339934) q[12];
rz(2.7908156670011537) q[12];
ry(3.141198745369578) q[13];
rz(-2.376171045651984) q[13];
ry(0.00019579441590812847) q[14];
rz(-1.39002441470251) q[14];
ry(-0.7912259907662511) q[15];
rz(0.3357815395240369) q[15];
ry(-0.7535706965649398) q[16];
rz(-2.2351171794684417) q[16];
ry(-0.22802087936851656) q[17];
rz(1.6223724402735442) q[17];
ry(-2.7699303614530852) q[18];
rz(-2.2034040911125903) q[18];
ry(1.7251614187454447) q[19];
rz(2.5834066177480324) q[19];
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
ry(-0.35550332556943154) q[0];
rz(0.8604521956868465) q[0];
ry(-1.2588363112612952) q[1];
rz(1.1634157337790778) q[1];
ry(-0.8376204850828232) q[2];
rz(-0.44220203079329856) q[2];
ry(0.1918868548441246) q[3];
rz(2.59837078025885) q[3];
ry(-1.5706675397582284) q[4];
rz(2.7601394637710435) q[4];
ry(3.1322221142875892) q[5];
rz(-0.14975853874002265) q[5];
ry(0.005845224371634039) q[6];
rz(2.6350051934520633) q[6];
ry(-2.784499370979139) q[7];
rz(1.5831471308984673) q[7];
ry(-3.007270948117616) q[8];
rz(-0.005024611985219165) q[8];
ry(-0.0005330980082760561) q[9];
rz(0.7290283159812551) q[9];
ry(-3.1090524577320893) q[10];
rz(0.3394504339965776) q[10];
ry(-0.6719846534188072) q[11];
rz(0.12432809339036753) q[11];
ry(1.1264297568364077) q[12];
rz(-1.4229278416213205) q[12];
ry(3.141293264935465) q[13];
rz(0.7946336410761603) q[13];
ry(-0.007508423057386082) q[14];
rz(0.9614609610516894) q[14];
ry(-1.2531134875923047) q[15];
rz(2.96971749916946) q[15];
ry(0.4273630370576411) q[16];
rz(-0.23063707767284555) q[16];
ry(-2.7530121154622003) q[17];
rz(0.8802524057383989) q[17];
ry(1.9451181823634416) q[18];
rz(1.9712313691760528) q[18];
ry(1.3428435428090326) q[19];
rz(1.264508033106117) q[19];
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
ry(-1.2105202971389382) q[0];
rz(-2.0548346695958477) q[0];
ry(-0.08003566995634534) q[1];
rz(-0.07564689733167233) q[1];
ry(2.2659050933353466) q[2];
rz(1.0916806374796388) q[2];
ry(2.5466156126965256) q[3];
rz(1.9371286665801468) q[3];
ry(2.6988969316769937) q[4];
rz(2.7728921328622067) q[4];
ry(1.5845686313524077) q[5];
rz(-2.7787709854743707) q[5];
ry(1.6108500212022885) q[6];
rz(0.7960936406631305) q[6];
ry(-1.5759388458302768) q[7];
rz(1.933664350395961) q[7];
ry(-1.824829308805239) q[8];
rz(3.092033722973909) q[8];
ry(3.1396524166847506) q[9];
rz(1.9980060404922348) q[9];
ry(3.0191399503081136) q[10];
rz(-2.871773356988928) q[10];
ry(-3.0892110959273076) q[11];
rz(-2.9100344416531105) q[11];
ry(0.032506727829108684) q[12];
rz(-1.1850996631851194) q[12];
ry(3.1415446547566623) q[13];
rz(1.422344330589259) q[13];
ry(-3.140734726455914) q[14];
rz(-3.1060418274316435) q[14];
ry(1.8125442444331352) q[15];
rz(0.9600878840981019) q[15];
ry(0.5193828474718379) q[16];
rz(3.100050900140966) q[16];
ry(-3.1346945033511227) q[17];
rz(-1.4075039058368506) q[17];
ry(2.5393668222817807) q[18];
rz(0.22898829030134668) q[18];
ry(0.5386549531323057) q[19];
rz(-2.092352469009106) q[19];
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
ry(1.7461503563815484) q[0];
rz(-0.5324987103231975) q[0];
ry(-1.4777614240958545) q[1];
rz(-1.3098084562715566) q[1];
ry(-0.5800423878346118) q[2];
rz(-1.0611739819337416) q[2];
ry(3.121430198180953) q[3];
rz(0.48187741030968784) q[3];
ry(-2.981420974151375) q[4];
rz(-2.734212576449253) q[4];
ry(3.1409480852275364) q[5];
rz(-2.928788917848076) q[5];
ry(-0.0004021245229894888) q[6];
rz(-0.6885618271405527) q[6];
ry(-3.14134710541347) q[7];
rz(-0.22482829320616304) q[7];
ry(1.4672330219943053) q[8];
rz(-1.6092335363645667) q[8];
ry(1.5640287352496065) q[9];
rz(-1.5747894232633526) q[9];
ry(0.01934681857542503) q[10];
rz(1.3381640547631157) q[10];
ry(0.7963883320867406) q[11];
rz(-2.213798179064353) q[11];
ry(0.2715043758853058) q[12];
rz(-2.667770170190694) q[12];
ry(0.0011410449116251444) q[13];
rz(1.5667494110166755) q[13];
ry(-0.0024128337039105914) q[14];
rz(-1.5000273645160487) q[14];
ry(-1.0281306414395148) q[15];
rz(-2.8656318023126572) q[15];
ry(2.788159453972887) q[16];
rz(-0.3941274036834397) q[16];
ry(-2.8866271833481223) q[17];
rz(0.75955587016217) q[17];
ry(-1.435637930703865) q[18];
rz(1.6369714247461693) q[18];
ry(2.713556870698549) q[19];
rz(0.5594376280775863) q[19];
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
ry(2.1688221479235628) q[0];
rz(1.1527351006041124) q[0];
ry(-2.0770781231799846) q[1];
rz(-2.051058584422708) q[1];
ry(-0.9989196476159492) q[2];
rz(0.5572800924037117) q[2];
ry(-2.2181508963903913) q[3];
rz(1.9758346414258021) q[3];
ry(2.583990191719974) q[4];
rz(-2.53514321531437) q[4];
ry(1.7044936077898933) q[5];
rz(-2.4404309831426367) q[5];
ry(2.8998360411658863) q[6];
rz(-1.2245700802653863) q[6];
ry(3.1390940752389396) q[7];
rz(-0.7759442661554976) q[7];
ry(-3.132494300576777) q[8];
rz(0.10829457650948804) q[8];
ry(-2.745676233438387) q[9];
rz(-1.5538854026613993) q[9];
ry(3.1299170494697903) q[10];
rz(3.12991608198667) q[10];
ry(1.4782288774091001) q[11];
rz(3.1248808088335127) q[11];
ry(1.222857076680497) q[12];
rz(-1.629254320929878) q[12];
ry(0.0005311482588279404) q[13];
rz(2.2882865587283527) q[13];
ry(-0.00021617578757070433) q[14];
rz(-0.6092224434827213) q[14];
ry(0.05243024302915215) q[15];
rz(-2.239805876670892) q[15];
ry(-3.1307888321919886) q[16];
rz(1.8758274855018129) q[16];
ry(0.0977096051794959) q[17];
rz(-3.1358194364132204) q[17];
ry(0.7290075641768716) q[18];
rz(1.3217792678179405) q[18];
ry(0.719017321343965) q[19];
rz(3.0224745809410756) q[19];
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
ry(-2.727303253659801) q[0];
rz(-2.2499372497511327) q[0];
ry(-1.191837850597894) q[1];
rz(0.6188833959924916) q[1];
ry(1.5039181703542837) q[2];
rz(2.566003678899304) q[2];
ry(-0.2417955885683707) q[3];
rz(-2.032933077799957) q[3];
ry(3.125919013578122) q[4];
rz(-0.17006108895401825) q[4];
ry(2.6347942429616467e-05) q[5];
rz(1.004369220603487) q[5];
ry(-0.0007765769049514759) q[6];
rz(1.29786064232289) q[6];
ry(3.1392998481372723) q[7];
rz(-2.754683533964034) q[7];
ry(-1.6005959896382382) q[8];
rz(1.6668024101441385) q[8];
ry(0.06512798412321619) q[9];
rz(1.550542401834501) q[9];
ry(-0.0009027573874424987) q[10];
rz(1.0211857638343205) q[10];
ry(-0.10165062863323517) q[11];
rz(0.018800149067153882) q[11];
ry(1.5362580124976744) q[12];
rz(-2.8222961407749914) q[12];
ry(-0.00031179673598824564) q[13];
rz(1.6870241960697347) q[13];
ry(0.0006350007958756796) q[14];
rz(-1.8992450721250589) q[14];
ry(3.021054604114371) q[15];
rz(1.7130830948578577) q[15];
ry(1.7722515835333175) q[16];
rz(-0.36012499875848397) q[16];
ry(-3.110720619484352) q[17];
rz(0.874229334577303) q[17];
ry(-3.0368519451834217) q[18];
rz(-1.815181542157231) q[18];
ry(0.3863939266846553) q[19];
rz(-2.1150888732569206) q[19];
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
ry(-0.7035436518077001) q[0];
rz(-1.8623619610505286) q[0];
ry(0.1630040954713481) q[1];
rz(-0.5998587015945164) q[1];
ry(1.7247321091039813) q[2];
rz(3.1174167694518133) q[2];
ry(-2.86699650273247) q[3];
rz(-1.4749916584321368) q[3];
ry(-2.6682393510876525) q[4];
rz(-2.6891875755449646) q[4];
ry(-0.8635867869383311) q[5];
rz(0.5540392771538922) q[5];
ry(3.1300635108108374) q[6];
rz(3.06501372956108) q[6];
ry(-0.0037062096190298988) q[7];
rz(-1.8887203242031618) q[7];
ry(2.5250802050498553) q[8];
rz(0.007957527651730523) q[8];
ry(-1.5211353151456037) q[9];
rz(0.0010995466557543224) q[9];
ry(-3.1316634240854158) q[10];
rz(-1.104568442877433) q[10];
ry(-1.661615118143053) q[11];
rz(0.817074661393357) q[11];
ry(1.0160937556110463) q[12];
rz(-3.113816022859717) q[12];
ry(2.888727451925061) q[13];
rz(-1.7701689644233038) q[13];
ry(-0.057126314803354686) q[14];
rz(-0.43448701696630554) q[14];
ry(0.22468703522684236) q[15];
rz(-0.6016564140644851) q[15];
ry(-2.8979284019479077) q[16];
rz(-0.572273286843008) q[16];
ry(3.0418354285933074) q[17];
rz(1.8740751134855502) q[17];
ry(-1.607605607413232) q[18];
rz(-2.1397909964220663) q[18];
ry(-0.5169701355333918) q[19];
rz(-2.4277314423459644) q[19];
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
ry(-1.3360617218356738) q[0];
rz(-1.0567674446083393) q[0];
ry(2.0924699490845926) q[1];
rz(1.3688987601637237) q[1];
ry(-1.5146041444318834) q[2];
rz(0.9760473613001279) q[2];
ry(0.04701214726502201) q[3];
rz(2.3832257513883732) q[3];
ry(-3.1378416662522306) q[4];
rz(2.308918376074323) q[4];
ry(3.140282173143512) q[5];
rz(-0.3136546770381523) q[5];
ry(-3.140329109773353) q[6];
rz(0.10836194720430126) q[6];
ry(0.003993204750203816) q[7];
rz(2.377098330190365) q[7];
ry(1.5517270618655319) q[8];
rz(-0.43762081810765174) q[8];
ry(-2.3697074122899755) q[9];
rz(3.1236357026689103) q[9];
ry(-0.0013307697054596446) q[10];
rz(0.6522751043638897) q[10];
ry(-3.1135103451514388) q[11];
rz(-1.4920130029380478) q[11];
ry(-3.136758370215281) q[12];
rz(-1.2961109552034191) q[12];
ry(-3.1413188669694776) q[13];
rz(-1.6660050736254826) q[13];
ry(-1.5700253041444643) q[14];
rz(2.9722823939571343) q[14];
ry(3.136777027909483) q[15];
rz(-0.7714174803454387) q[15];
ry(1.551562108523307) q[16];
rz(-1.2940478621980054) q[16];
ry(2.6386666384273547) q[17];
rz(-2.6553411858686293) q[17];
ry(-0.0715737978084358) q[18];
rz(-2.652721767616469) q[18];
ry(1.9735566481793976) q[19];
rz(-2.776780100150248) q[19];
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
ry(1.6153228934050516) q[0];
rz(1.1315123096955733) q[0];
ry(0.6727058432042723) q[1];
rz(0.2539098838674161) q[1];
ry(-1.6129892781268804) q[2];
rz(0.9156330386939694) q[2];
ry(-1.943710851333847) q[3];
rz(2.3662540087460764) q[3];
ry(-1.4769973471162956) q[4];
rz(0.6353600494337819) q[4];
ry(1.1763104530663204) q[5];
rz(2.3530599815205657) q[5];
ry(3.073021634026757) q[6];
rz(2.4962449806180342) q[6];
ry(-0.002327735482118065) q[7];
rz(-3.0114665806334737) q[7];
ry(-1.1396193953815263) q[8];
rz(-1.3789088944434758) q[8];
ry(3.015310371652514) q[9];
rz(-0.01916124433725913) q[9];
ry(-1.5813998533424565) q[10];
rz(-0.9528871993652523) q[10];
ry(0.0029524468212070824) q[11];
rz(0.8844259246536028) q[11];
ry(1.5746965046971733) q[12];
rz(1.1271924928711776) q[12];
ry(-1.8652379127312635) q[13];
rz(0.02154830012863584) q[13];
ry(2.780591812907373) q[14];
rz(2.5881740145344145) q[14];
ry(-3.1414316038270917) q[15];
rz(1.5869149708499097) q[15];
ry(3.140542436049354) q[16];
rz(-0.2309435806820801) q[16];
ry(2.8639705538678473) q[17];
rz(0.8663822236537164) q[17];
ry(-2.0096571449717366) q[18];
rz(1.0757766442032874) q[18];
ry(0.3757027114795212) q[19];
rz(0.5807554842858451) q[19];
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
ry(-1.5630729340633343) q[0];
rz(-0.1843773303055909) q[0];
ry(2.219105525588197) q[1];
rz(0.09364799321772567) q[1];
ry(-0.6018734081079998) q[2];
rz(-1.1276454856970064) q[2];
ry(-3.128736761268366) q[3];
rz(0.9351738327955506) q[3];
ry(-0.03232581400866241) q[4];
rz(-0.830628193993819) q[4];
ry(-3.1398360654092583) q[5];
rz(-1.4911393581607044) q[5];
ry(-3.140953979881606) q[6];
rz(-0.9280342196280683) q[6];
ry(-0.00015441779060676453) q[7];
rz(2.6242498886600654) q[7];
ry(-1.5709286735051953) q[8];
rz(-0.2634891291566068) q[8];
ry(-1.1841148001546165) q[9];
rz(0.09574198560847808) q[9];
ry(-0.00010763269145742612) q[10];
rz(0.945655594491418) q[10];
ry(-3.1413292604445715) q[11];
rz(2.4369891439489795) q[11];
ry(0.09569724866620634) q[12];
rz(2.623931171248214) q[12];
ry(0.3841553314016233) q[13];
rz(0.004701363523839319) q[13];
ry(0.010491461064773056) q[14];
rz(0.33896798462524136) q[14];
ry(0.0008872230623070369) q[15];
rz(-1.134526403886034) q[15];
ry(0.006492670909913123) q[16];
rz(0.8519755632803694) q[16];
ry(-2.0543820901081045) q[17];
rz(-2.420625586746779) q[17];
ry(2.2286812823918485) q[18];
rz(0.9410629232495662) q[18];
ry(2.230332695514269) q[19];
rz(-0.6525100578287669) q[19];
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
ry(1.847081807182897) q[0];
rz(0.20749287729516983) q[0];
ry(1.4093371413423195) q[1];
rz(-0.9802956668338226) q[1];
ry(1.7088548246065276) q[2];
rz(1.7944432288813985) q[2];
ry(0.8842959052788313) q[3];
rz(-2.987519281486771) q[3];
ry(2.0121817962363635) q[4];
rz(3.0876076106930093) q[4];
ry(-1.4784463901956872) q[5];
rz(0.9453918467258723) q[5];
ry(0.19457481684700806) q[6];
rz(0.26730756420660734) q[6];
ry(-1.6153738979859282) q[7];
rz(1.539382060963776) q[7];
ry(1.3396136021706324) q[8];
rz(-0.5798501365546118) q[8];
ry(-0.0016346991303113059) q[9];
rz(0.04780041207884623) q[9];
ry(-1.5802773187671946) q[10];
rz(1.0766225518431944) q[10];
ry(1.5270814687640764) q[11];
rz(-3.141580452916314) q[11];
ry(0.002510456057042809) q[12];
rz(1.7517615158312507) q[12];
ry(1.527380154821886) q[13];
rz(-1.5773385074833612) q[13];
ry(-2.2246940323704743) q[14];
rz(0.032069296112127256) q[14];
ry(-3.1404130664758636) q[15];
rz(1.7363464287891324) q[15];
ry(-0.003983831819610373) q[16];
rz(2.538282557151216) q[16];
ry(-2.481582344368047) q[17];
rz(2.848334063619369) q[17];
ry(3.0137352456419104) q[18];
rz(-2.261470393609429) q[18];
ry(-1.063839915418698) q[19];
rz(-2.7666106306925657) q[19];
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
ry(2.476310808726687) q[0];
rz(2.8738575149562218) q[0];
ry(1.6962942615728644) q[1];
rz(1.6837276629605) q[1];
ry(2.1502488680184983) q[2];
rz(-0.020730182216834866) q[2];
ry(-2.9474209754729204) q[3];
rz(-2.4820043244123813) q[3];
ry(0.005771473502218072) q[4];
rz(0.40362517778699214) q[4];
ry(3.1404620916335206) q[5];
rz(-2.0595698630754087) q[5];
ry(3.124510140703047) q[6];
rz(-1.372355235179472) q[6];
ry(3.1409818525680944) q[7];
rz(-0.3065211070873035) q[7];
ry(-0.014412543493483199) q[8];
rz(-1.2312306568839255) q[8];
ry(3.1190087050957067) q[9];
rz(-2.9988914339275383) q[9];
ry(-3.1402979070764108) q[10];
rz(1.0732563197167613) q[10];
ry(1.5718901315810516) q[11];
rz(-1.5723812239105595) q[11];
ry(3.1414565029331816) q[12];
rz(-0.13080686959920926) q[12];
ry(-0.1373092316146645) q[13];
rz(2.8361469783382645) q[13];
ry(3.1354128030456536) q[14];
rz(0.11822149988623458) q[14];
ry(3.1380413980402855) q[15];
rz(2.3646661425051505) q[15];
ry(3.1378213955199237) q[16];
rz(-0.24629259444429155) q[16];
ry(-0.1546258732501613) q[17];
rz(1.9818666618248102) q[17];
ry(1.994912404118529) q[18];
rz(-0.5765638044574803) q[18];
ry(2.11609137250002) q[19];
rz(-1.632622578994295) q[19];
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
ry(-1.7305594524955483) q[0];
rz(-0.8834582224187271) q[0];
ry(-1.3345570215315572) q[1];
rz(3.1228432926067318) q[1];
ry(-1.0853976312533844) q[2];
rz(-2.273971190185639) q[2];
ry(0.10313830457812087) q[3];
rz(-0.2050590330438737) q[3];
ry(1.4246043724426585) q[4];
rz(-0.7496579252325384) q[4];
ry(0.8246702527823289) q[5];
rz(2.2216312075228233) q[5];
ry(-1.6035693894070195) q[6];
rz(-1.6860545051240134) q[6];
ry(-1.7537823271038895) q[7];
rz(-0.8068988498106346) q[7];
ry(-2.857619847282252) q[8];
rz(-2.741702488624864) q[8];
ry(-1.8874074942313799) q[9];
rz(0.00046314865592207575) q[9];
ry(1.5730601678896265) q[10];
rz(1.12893678283234) q[10];
ry(1.5691964378632781) q[11];
rz(1.6171372713399434) q[11];
ry(-0.0016680488481473966) q[12];
rz(0.07064089540385866) q[12];
ry(0.0004051761161747436) q[13];
rz(-1.2735251715497227) q[13];
ry(-0.6878833255683814) q[14];
rz(1.611872726302158) q[14];
ry(-3.140258185600175) q[15];
rz(-1.0352158441583355) q[15];
ry(-1.5736842017423394) q[16];
rz(-0.0011813656878496557) q[16];
ry(-1.4303760377728827) q[17];
rz(1.7478683771985082) q[17];
ry(-2.0102920452980655) q[18];
rz(3.1015502189497917) q[18];
ry(2.87768701197724) q[19];
rz(2.860586444693432) q[19];
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
ry(1.1194877945648154) q[0];
rz(-2.5164775759769253) q[0];
ry(1.913232413080854) q[1];
rz(2.9885516302478923) q[1];
ry(-0.517260145509943) q[2];
rz(-1.635708657391752) q[2];
ry(-0.055055631120778645) q[3];
rz(-1.1705863485982881) q[3];
ry(-2.107431867481427) q[4];
rz(0.002898087593923506) q[4];
ry(3.1354930150440654) q[5];
rz(0.40671615576707065) q[5];
ry(0.003770130381356651) q[6];
rz(-3.01696587262257) q[6];
ry(0.0027473145337991943) q[7];
rz(0.7200682061306246) q[7];
ry(0.030104609055890737) q[8];
rz(-1.852938987066631) q[8];
ry(1.5684264965153938) q[9];
rz(-0.01162646260748447) q[9];
ry(-9.86749596466811e-05) q[10];
rz(0.7814532191986129) q[10];
ry(0.7582073066456703) q[11];
rz(-0.0011089704169915976) q[11];
ry(-1.459500543583097) q[12];
rz(-0.14519094268878874) q[12];
ry(-1.5811550451481138) q[13];
rz(0.032248524589332875) q[13];
ry(0.020417200883979802) q[14];
rz(1.5153755330324592) q[14];
ry(3.140570504391552) q[15];
rz(-1.2545113000397663) q[15];
ry(-2.709023333899749) q[16];
rz(-1.5632913025915791) q[16];
ry(1.5729290658250221) q[17];
rz(-1.571347179399472) q[17];
ry(1.5698950066172832) q[18];
rz(1.3513597007538607) q[18];
ry(-1.5328094666513232) q[19];
rz(0.8858199154492667) q[19];
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
ry(-0.5264171549183291) q[0];
rz(1.9378269734012985) q[0];
ry(-1.2025320606958116) q[1];
rz(1.7049717918160765) q[1];
ry(3.117091463452105) q[2];
rz(0.8239352452355069) q[2];
ry(-2.7445970369960846) q[3];
rz(1.3007552664730344) q[3];
ry(0.13573032162438065) q[4];
rz(-0.011196267978569562) q[4];
ry(0.009181800514427962) q[5];
rz(-0.9421857008065307) q[5];
ry(-0.036830867366199324) q[6];
rz(0.5840682268270798) q[6];
ry(1.7421662545135401) q[7];
rz(0.03134850729732164) q[7];
ry(-3.1405568406578617) q[8];
rz(1.8391748607103722) q[8];
ry(0.653488946776458) q[9];
rz(1.6054198610326578) q[9];
ry(-3.1384607984021766) q[10];
rz(1.885672263858071) q[10];
ry(1.5686717476690675) q[11];
rz(-0.002394339549198179) q[11];
ry(3.1414086554141147) q[12];
rz(-0.1500477375593354) q[12];
ry(0.02427845757192681) q[13];
rz(3.1088025883544934) q[13];
ry(-1.8724701903846956) q[14];
rz(-2.774995333688016) q[14];
ry(-3.141492557768885) q[15];
rz(-0.5821160614684642) q[15];
ry(1.5971811421150588) q[16];
rz(1.5665851416744738) q[16];
ry(1.574278908840192) q[17];
rz(2.5514419697790784) q[17];
ry(-3.139455436045022) q[18];
rz(-0.20870040114311256) q[18];
ry(1.71561424693627) q[19];
rz(-2.7907594753472855) q[19];
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
ry(-0.9407274504831723) q[0];
rz(-1.7605535945395006) q[0];
ry(-2.436248594705742) q[1];
rz(2.126724723360618) q[1];
ry(2.0423067449328647) q[2];
rz(2.8871400643336966) q[2];
ry(3.083772599062132) q[3];
rz(1.7934546891254763) q[3];
ry(1.047538175323215) q[4];
rz(0.5847660587738723) q[4];
ry(0.0072917981549846) q[5];
rz(-0.8599212300447993) q[5];
ry(3.13693173397985) q[6];
rz(-2.790197466599102) q[6];
ry(3.138507524570864) q[7];
rz(-1.5305824699919146) q[7];
ry(3.016671057854448) q[8];
rz(1.5209339013945422) q[8];
ry(-3.0079380945142264) q[9];
rz(1.5938853822506598) q[9];
ry(3.1415719540790135) q[10];
rz(-0.6301244095555454) q[10];
ry(0.7740580816848929) q[11];
rz(-1.6001095194492962) q[11];
ry(2.7658166929708328) q[12];
rz(1.661845450939583) q[12];
ry(1.5940388060944253) q[13];
rz(-1.5707314862697206) q[13];
ry(-3.1405644488216398) q[14];
rz(-2.776519132600692) q[14];
ry(-0.0003565912228742363) q[15];
rz(-0.49238032642798457) q[15];
ry(-1.570915736541897) q[16];
rz(3.0500915933258326) q[16];
ry(-3.1314479955784917) q[17];
rz(-1.0614708689471617) q[17];
ry(0.009596685431239038) q[18];
rz(-0.010024170197756584) q[18];
ry(-3.13830655451214) q[19];
rz(1.2996781954804455) q[19];
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
ry(-1.7871494957324334) q[0];
rz(-1.7976732464050782) q[0];
ry(-0.47868127340898686) q[1];
rz(2.3027077893079246) q[1];
ry(0.733275793481817) q[2];
rz(0.923167628135495) q[2];
ry(1.1660575561214233) q[3];
rz(-1.14109399802709) q[3];
ry(2.9746331713817096) q[4];
rz(0.5822305063739845) q[4];
ry(-2.3659423037754) q[5];
rz(-0.9762121543405762) q[5];
ry(-1.5629400106452755) q[6];
rz(-2.009650725177943) q[6];
ry(-1.593499072874909) q[7];
rz(-1.7746963871379478) q[7];
ry(-3.1408921119185726) q[8];
rz(-2.2646640535536156) q[8];
ry(1.557349738962988) q[9];
rz(-1.4017460378904043) q[9];
ry(-3.12979557055978) q[10];
rz(-2.23367782024462) q[10];
ry(-1.567333851476115) q[11];
rz(-2.536305730624859) q[11];
ry(-0.0025670809171990444) q[12];
rz(1.5438005887400488) q[12];
ry(1.5603866324725297) q[13];
rz(3.0957334858964782) q[13];
ry(1.5657695148726472) q[14];
rz(-1.4307139412157586) q[14];
ry(1.5717841614101407) q[15];
rz(0.0015008511711585015) q[15];
ry(0.010185942495184081) q[16];
rz(0.5826280994282157) q[16];
ry(3.139595852372313) q[17];
rz(2.5644970567233023) q[17];
ry(2.920812238182578) q[18];
rz(0.0013197514321256838) q[18];
ry(-1.7120736924915976) q[19];
rz(-2.0486117121346066) q[19];
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
ry(1.4634413793037364) q[0];
rz(1.872805591478033) q[0];
ry(-2.752695843326914) q[1];
rz(-0.3387488281202208) q[1];
ry(0.40539715354269656) q[2];
rz(-1.2936839145362056) q[2];
ry(-0.0016705407817232398) q[3];
rz(-1.0128540850720527) q[3];
ry(-0.009165913838940732) q[4];
rz(-2.908367673095253) q[4];
ry(-0.009974236163111883) q[5];
rz(-0.9612690868946556) q[5];
ry(3.1410471859027873) q[6];
rz(2.5762464578682884) q[6];
ry(3.0552168517542873) q[7];
rz(0.4209519278586855) q[7];
ry(0.00013416558358840373) q[8];
rz(2.9597546603118667) q[8];
ry(0.05712387036693975) q[9];
rz(-1.7353831721810966) q[9];
ry(-3.141234395106051) q[10];
rz(-1.3952593859812543) q[10];
ry(-0.000514823174234889) q[11];
rz(-0.6158561413579908) q[11];
ry(0.0001930875023106182) q[12];
rz(1.5004650981796475) q[12];
ry(-2.9091291396893686) q[13];
rz(3.047424392695732) q[13];
ry(-3.141526846475416) q[14];
rz(1.5652637355726353) q[14];
ry(-2.967941607000073) q[15];
rz(1.5468625746002616) q[15];
ry(2.8051633256249144) q[16];
rz(2.2700839550259797) q[16];
ry(1.5707786152361343) q[17];
rz(-1.5984231328266825) q[17];
ry(1.5640198825014657) q[18];
rz(1.569787714428417) q[18];
ry(3.1414895802496696) q[19];
rz(2.6300263171465565) q[19];
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
ry(-2.272043820776248) q[0];
rz(1.140156514781489) q[0];
ry(-1.6916775873132415) q[1];
rz(0.20342261348812285) q[1];
ry(-1.8606922330852713) q[2];
rz(-1.2739169690310517) q[2];
ry(0.7513910942359101) q[3];
rz(-0.5160656521027357) q[3];
ry(-2.986225618000229) q[4];
rz(1.0760202585112415) q[4];
ry(0.3769733477360182) q[5];
rz(-2.5686607481614248) q[5];
ry(-3.088029204896474) q[6];
rz(-1.1566680831967375) q[6];
ry(0.011301338661699134) q[7];
rz(-0.06159978063530425) q[7];
ry(0.0005591496922088268) q[8];
rz(1.491115201836716) q[8];
ry(-1.8022673048305082) q[9];
rz(0.4161142580778634) q[9];
ry(3.1339172414970995) q[10];
rz(-0.8177150640046177) q[10];
ry(1.5704940839058894) q[11];
rz(1.9700672102084487) q[11];
ry(-3.1388571996025068) q[12];
rz(-1.053942175774523) q[12];
ry(-1.581194676314194) q[13];
rz(1.981269797668419) q[13];
ry(-3.1399988828638317) q[14];
rz(-2.7087137389821008) q[14];
ry(-3.141087601216854) q[15];
rz(0.3832677934034425) q[15];
ry(0.00024090493994588513) q[16];
rz(-1.254476211128738) q[16];
ry(-0.06938000098071394) q[17];
rz(0.44011122160851457) q[17];
ry(-1.5706430788910533) q[18];
rz(-1.2338210958149882) q[18];
ry(-1.5700649172118428) q[19];
rz(0.4122377787242413) q[19];