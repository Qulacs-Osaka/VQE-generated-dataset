OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.08219692396319359) q[0];
rz(-1.9378793879336351) q[0];
ry(0.16643538849695894) q[1];
rz(-0.39327078067477217) q[1];
ry(0.0031582109583361273) q[2];
rz(-1.8257320914976436) q[2];
ry(0.293636466448355) q[3];
rz(1.3764927393893764) q[3];
ry(-2.866546802407642) q[4];
rz(1.0724799111483962) q[4];
ry(-3.1357605760561458) q[5];
rz(-0.7215272929721349) q[5];
ry(1.0041959055160212) q[6];
rz(0.0016119383702085585) q[6];
ry(0.008153658157646682) q[7];
rz(-0.03859106622943731) q[7];
ry(-0.18662363906782442) q[8];
rz(1.4608432339100215) q[8];
ry(-0.06586212758247956) q[9];
rz(2.3844632482301558) q[9];
ry(-3.105435474602253) q[10];
rz(0.21809567419740833) q[10];
ry(-1.781820493726813) q[11];
rz(0.3691172642716882) q[11];
ry(2.3125695640906536) q[12];
rz(-0.24809522816595386) q[12];
ry(1.40013626986236) q[13];
rz(2.810191382418119) q[13];
ry(-0.002189444579282295) q[14];
rz(1.0736702077805196) q[14];
ry(-2.874269223123895) q[15];
rz(1.7789623754260575) q[15];
ry(1.744126161975565) q[16];
rz(-1.9789379481009917) q[16];
ry(-1.0789025205870175) q[17];
rz(-1.7237983274503823) q[17];
ry(3.135508833339796) q[18];
rz(2.680607333510051) q[18];
ry(0.4758927148891754) q[19];
rz(-0.2137425627009685) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.1722106823012335) q[0];
rz(-3.125032961122289) q[0];
ry(3.107035859560544) q[1];
rz(-1.5110492828515758) q[1];
ry(3.138428779175015) q[2];
rz(-0.8525941517955165) q[2];
ry(-2.929713758603956) q[3];
rz(0.1629597992883962) q[3];
ry(-2.9660243970500475) q[4];
rz(1.6003596520128953) q[4];
ry(1.6554718868859526) q[5];
rz(1.9154289314276307) q[5];
ry(-2.1296760526173335) q[6];
rz(0.5199321584946219) q[6];
ry(-1.552501186468371) q[7];
rz(-3.014002252057686) q[7];
ry(2.5699198765631768) q[8];
rz(-2.1981060018627963) q[8];
ry(-3.097035327055829) q[9];
rz(0.9568864922245893) q[9];
ry(-3.087199367139977) q[10];
rz(-2.571558081379216) q[10];
ry(1.426019862228726) q[11];
rz(-0.631508635989636) q[11];
ry(2.832954094043524) q[12];
rz(-3.03809397985585) q[12];
ry(-1.6508482036232766) q[13];
rz(-1.0115007083635734) q[13];
ry(-0.0016319863996072821) q[14];
rz(-1.934633469218651) q[14];
ry(-2.9514961835583913) q[15];
rz(-2.653575752521629) q[15];
ry(0.1625391254335844) q[16];
rz(1.9554183654795207) q[16];
ry(-0.30427695311698244) q[17];
rz(1.9086456255439082) q[17];
ry(3.1116765729649463) q[18];
rz(2.827182023315182) q[18];
ry(2.7834223425644224) q[19];
rz(-2.241452556816203) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.8325784125936078) q[0];
rz(3.10740853832588) q[0];
ry(-0.13126929450194247) q[1];
rz(-1.5121957750175996) q[1];
ry(-0.7542111044246251) q[2];
rz(0.4991776433240975) q[2];
ry(0.5421013070583856) q[3];
rz(0.6399994723758585) q[3];
ry(-1.623233729320832) q[4];
rz(0.37977817741069764) q[4];
ry(-1.7307264036565624) q[5];
rz(-2.0856182377532093) q[5];
ry(3.11717195443961) q[6];
rz(2.3001943530665625) q[6];
ry(-1.267766610514334) q[7];
rz(0.489507647975812) q[7];
ry(-0.508847611404593) q[8];
rz(0.20821332995198724) q[8];
ry(-0.027660973188921925) q[9];
rz(-1.2636614013342569) q[9];
ry(0.004932106576845285) q[10];
rz(-2.844081105329592) q[10];
ry(-2.9756754041574736) q[11];
rz(2.4686077905142856) q[11];
ry(-3.056056394625356) q[12];
rz(3.0724803798616427) q[12];
ry(-2.601543118551715) q[13];
rz(-3.0381324714821565) q[13];
ry(-0.0018276915035398314) q[14];
rz(-0.7687810708518663) q[14];
ry(2.6345470997865066) q[15];
rz(-0.4163331511852102) q[15];
ry(-1.802713569656488) q[16];
rz(0.9861484340954058) q[16];
ry(-0.7808642914339561) q[17];
rz(3.0006981656214227) q[17];
ry(-0.011139629105410059) q[18];
rz(2.808200948433654) q[18];
ry(-0.03606339551631255) q[19];
rz(-0.9071350804020293) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.7768321176769071) q[0];
rz(1.7182885729517658) q[0];
ry(1.2599532556477682) q[1];
rz(0.7592758514777997) q[1];
ry(-0.0006899032576930253) q[2];
rz(-2.952787923615626) q[2];
ry(9.496314004842077e-05) q[3];
rz(0.00976651339832557) q[3];
ry(0.0012191110822810435) q[4];
rz(-3.0437297308553477) q[4];
ry(-0.3971681785265764) q[5];
rz(-2.448985770667921) q[5];
ry(0.005434213454297117) q[6];
rz(-1.5659572173057457) q[6];
ry(2.912136390705859) q[7];
rz(-3.021806167508026) q[7];
ry(2.875235917655571) q[8];
rz(1.2554302255165124) q[8];
ry(1.8457799681126728) q[9];
rz(-2.3563912586177542) q[9];
ry(-2.1811013701939936) q[10];
rz(-1.4697313930162421) q[10];
ry(1.7413782561858608) q[11];
rz(1.2719203522477693) q[11];
ry(-3.0782445205874374) q[12];
rz(-0.7618713938386502) q[12];
ry(-1.4919817441044114) q[13];
rz(2.0548486686609775) q[13];
ry(0.004862953062027204) q[14];
rz(-1.101173956379961) q[14];
ry(-0.19730705459445505) q[15];
rz(-2.437426112960611) q[15];
ry(2.63748483972008) q[16];
rz(2.4796006852263814) q[16];
ry(-1.3240898246407287) q[17];
rz(3.0008627104872234) q[17];
ry(2.5001496874695746) q[18];
rz(0.7761589379518785) q[18];
ry(2.2316640860406887) q[19];
rz(-0.7126298565763651) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.6786133663904732) q[0];
rz(-3.048845885724426) q[0];
ry(1.8341189771271815) q[1];
rz(2.7722976140986546) q[1];
ry(1.4098734243122173) q[2];
rz(-2.394691287347794) q[2];
ry(-0.5252080609931569) q[3];
rz(2.608185167411783) q[3];
ry(1.6826650252406272) q[4];
rz(-1.56310538469888) q[4];
ry(-1.152729748272466) q[5];
rz(2.5432461892506786) q[5];
ry(0.5806365120544729) q[6];
rz(-0.837248156071742) q[6];
ry(0.835744390245915) q[7];
rz(-2.870429476775423) q[7];
ry(-3.1228319462451943) q[8];
rz(0.9525569331926169) q[8];
ry(0.09397594937196274) q[9];
rz(2.1555050569753877) q[9];
ry(-3.139401676933602) q[10];
rz(-2.6808344862402973) q[10];
ry(3.026994862107091) q[11];
rz(0.962555686731772) q[11];
ry(-3.125387954858228) q[12];
rz(2.6150505560083097) q[12];
ry(0.03628761061781205) q[13];
rz(-0.4789343454027662) q[13];
ry(0.07167946118297504) q[14];
rz(0.6545410357776739) q[14];
ry(0.9153254947792515) q[15];
rz(0.9603058963947461) q[15];
ry(-2.815646207303937) q[16];
rz(-1.8833077259042046) q[16];
ry(0.12224813577910575) q[17];
rz(2.501756301340893) q[17];
ry(-2.8030174041174725) q[18];
rz(-1.4059037541726873) q[18];
ry(-0.28353213444030295) q[19];
rz(2.1885573305419967) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.5051821424186951) q[0];
rz(-1.8386235814608969) q[0];
ry(0.4262208535333929) q[1];
rz(-2.9996207972519975) q[1];
ry(-2.163755506825148) q[2];
rz(1.112013853731989) q[2];
ry(3.1378969163715302) q[3];
rz(-1.4944140241240271) q[3];
ry(-1.0949541499292512) q[4];
rz(0.11247082351645338) q[4];
ry(-0.4228911069098098) q[5];
rz(0.36883572838965506) q[5];
ry(-0.05292683965592087) q[6];
rz(1.4770716926589262) q[6];
ry(-0.07000135509554806) q[7];
rz(2.3938768870407277) q[7];
ry(1.6203910967677206) q[8];
rz(-0.5190635067419875) q[8];
ry(-1.0518378965199686) q[9];
rz(-0.9754043208649145) q[9];
ry(-2.9238861573785746) q[10];
rz(-0.02473221784608981) q[10];
ry(-2.9652176499831806) q[11];
rz(0.276204365959158) q[11];
ry(-2.7529441448724983) q[12];
rz(-2.3406758867178903) q[12];
ry(2.7535234047585244) q[13];
rz(1.440292903868932) q[13];
ry(-0.011742763889131065) q[14];
rz(1.959870458817253) q[14];
ry(-2.7580324524736293) q[15];
rz(2.740261010718647) q[15];
ry(-2.961353911829584) q[16];
rz(1.7094443906882246) q[16];
ry(0.8424182785967073) q[17];
rz(-2.7035274093847725) q[17];
ry(-0.48432782705241717) q[18];
rz(2.284369442642258) q[18];
ry(0.468691732299427) q[19];
rz(0.16815142542659253) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.47565707646518557) q[0];
rz(-0.007361779439929882) q[0];
ry(-0.9971954724670304) q[1];
rz(-2.5828665299521893) q[1];
ry(-0.5954020020253606) q[2];
rz(-3.0329355765874704) q[2];
ry(-3.140578848637998) q[3];
rz(1.2503316433799834) q[3];
ry(-0.3805753794533544) q[4];
rz(-2.9166233128986816) q[4];
ry(-2.826464636612137) q[5];
rz(-2.421082500018936) q[5];
ry(-2.5574703470807774) q[6];
rz(0.734534230759695) q[6];
ry(3.122244516954777) q[7];
rz(-1.8769202666184017) q[7];
ry(3.1301235539287164) q[8];
rz(-1.105193146048218) q[8];
ry(-0.7258270418680253) q[9];
rz(-1.4030760624132972) q[9];
ry(3.109232229658764) q[10];
rz(-0.5073518128521339) q[10];
ry(2.832270866708156) q[11];
rz(-0.5831275583439828) q[11];
ry(-1.6371605454492357) q[12];
rz(-0.4394380001000224) q[12];
ry(3.0036250501260735) q[13];
rz(0.3294524337349491) q[13];
ry(0.11614721582623218) q[14];
rz(-2.9481772576176724) q[14];
ry(1.5105874422104508) q[15];
rz(-2.433158748699789) q[15];
ry(-0.0003701340488535108) q[16];
rz(3.0583799545985) q[16];
ry(1.2232409495937457) q[17];
rz(-2.401242587087065) q[17];
ry(-2.8031606838340726) q[18];
rz(-2.0454600772388307) q[18];
ry(-1.4172467179920412) q[19];
rz(1.4338402523608167) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.813666967180981) q[0];
rz(-0.8542409997482672) q[0];
ry(2.399510954115897) q[1];
rz(0.3013079883283236) q[1];
ry(1.501960208926302) q[2];
rz(-1.6590760176309969) q[2];
ry(-1.4238424608189328) q[3];
rz(-0.5418331471568) q[3];
ry(-0.9151261137705609) q[4];
rz(-2.0805424437524582) q[4];
ry(0.772938242545671) q[5];
rz(0.42409657003095935) q[5];
ry(0.09207920716415252) q[6];
rz(-0.29911874677661926) q[6];
ry(-2.903215726667839) q[7];
rz(-1.051387476395563) q[7];
ry(-2.7625641051917342) q[8];
rz(-2.2468578622122184) q[8];
ry(-2.6077395601433673) q[9];
rz(-0.6136876226523539) q[9];
ry(1.1981961690730536) q[10];
rz(-2.104808191951906) q[10];
ry(-0.04218077292065957) q[11];
rz(-0.44121816630387684) q[11];
ry(-0.5074516611767099) q[12];
rz(-0.10412329936242594) q[12];
ry(-0.11382942884413794) q[13];
rz(-1.0478097747309303) q[13];
ry(-0.05160658819732511) q[14];
rz(0.37224523001908144) q[14];
ry(0.6989168396501683) q[15];
rz(-1.6896410643904138) q[15];
ry(-2.4421320380928093) q[16];
rz(-2.5498052065011616) q[16];
ry(-1.2565712936738458) q[17];
rz(1.5179846769479244) q[17];
ry(2.8313090650376833) q[18];
rz(-0.7146246194156671) q[18];
ry(0.24322025823247362) q[19];
rz(2.6012893371432835) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.8607388835631804) q[0];
rz(0.3059570445808877) q[0];
ry(1.6371424067063833) q[1];
rz(0.5394062751189139) q[1];
ry(1.9588584593145482) q[2];
rz(-0.7604217819685984) q[2];
ry(-2.9130965610881643) q[3];
rz(0.19953657829744265) q[3];
ry(3.138589400003934) q[4];
rz(-2.876695930740211) q[4];
ry(-0.2288472798846115) q[5];
rz(-2.3566852979531303) q[5];
ry(-2.655321935168745) q[6];
rz(1.6317011276109277) q[6];
ry(-0.17057439715660794) q[7];
rz(-3.132492824630834) q[7];
ry(-3.1415137195214733) q[8];
rz(-1.2275529223406405) q[8];
ry(3.085030097594773) q[9];
rz(0.009755056134277495) q[9];
ry(-3.106956191111368) q[10];
rz(-1.971393857462601) q[10];
ry(2.610691349990275) q[11];
rz(-2.8024882960038786) q[11];
ry(-0.6475701728832468) q[12];
rz(-3.007907087378621) q[12];
ry(-3.1068787093862986) q[13];
rz(1.9712754036190652) q[13];
ry(-3.117940451261716) q[14];
rz(1.6231533084376268) q[14];
ry(0.011270087414377109) q[15];
rz(0.6355244006850683) q[15];
ry(0.01015875302560783) q[16];
rz(2.486160417822255) q[16];
ry(-0.018299023146984617) q[17];
rz(0.02494801374079277) q[17];
ry(-1.7881209794544146) q[18];
rz(0.36824805656917464) q[18];
ry(2.242682874230832) q[19];
rz(-1.7632723013793696) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.330579640121293) q[0];
rz(0.9568865988249913) q[0];
ry(3.1280738936167714) q[1];
rz(1.5302770548863984) q[1];
ry(-0.0006757069521993111) q[2];
rz(2.0348060378353443) q[2];
ry(2.842172555543455) q[3];
rz(2.234233269830195) q[3];
ry(-3.0587929253614172) q[4];
rz(-1.7364398136236563) q[4];
ry(-1.9505909022913208) q[5];
rz(-0.9286469329464461) q[5];
ry(0.022704951276059027) q[6];
rz(2.9790037977153885) q[6];
ry(-0.09233197923416368) q[7];
rz(0.25096522327690735) q[7];
ry(2.235364707368387) q[8];
rz(0.7663609471924832) q[8];
ry(-0.18701569241010976) q[9];
rz(-2.5371776741036025) q[9];
ry(0.11861994657766513) q[10];
rz(2.02851330019433) q[10];
ry(3.0905928988253066) q[11];
rz(0.3392730560515727) q[11];
ry(-1.644953152250567) q[12];
rz(-0.5676449942971783) q[12];
ry(-1.2852886288156002) q[13];
rz(-1.5082049777102045) q[13];
ry(-0.07101281395847003) q[14];
rz(-2.0736201431714862) q[14];
ry(-2.4792292182249813) q[15];
rz(-2.2369783173809528) q[15];
ry(0.27073682158649137) q[16];
rz(2.613906464666975) q[16];
ry(-3.1110888037274016) q[17];
rz(-1.0966565554511476) q[17];
ry(-1.2075393402794) q[18];
rz(-0.02337189993604462) q[18];
ry(2.433086310114877) q[19];
rz(3.1391138024877714) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.374792397908032) q[0];
rz(1.4365595407257412) q[0];
ry(-1.4018267104869564) q[1];
rz(1.9515538608734637) q[1];
ry(-1.9084312517095459) q[2];
rz(1.2179580866724846) q[2];
ry(-2.7320266662176325) q[3];
rz(-2.6265324392803526) q[3];
ry(-0.6105961016748627) q[4];
rz(0.6304103698292858) q[4];
ry(0.8116155249628632) q[5];
rz(0.4559103052386684) q[5];
ry(-0.2974863125853147) q[6];
rz(2.5446684316816386) q[6];
ry(-0.27718628460353123) q[7];
rz(1.7125279003494724) q[7];
ry(0.02304717782578823) q[8];
rz(2.7403546589167913) q[8];
ry(-3.1141917841018647) q[9];
rz(-2.5277536717751814) q[9];
ry(3.132066257184185) q[10];
rz(-1.6604721433387546) q[10];
ry(-2.8719867190278365) q[11];
rz(-0.3926627408810308) q[11];
ry(-0.6514849834189581) q[12];
rz(-2.2271514896836666) q[12];
ry(-1.4637562235282378) q[13];
rz(-0.1998530856490136) q[13];
ry(-0.06126334660728894) q[14];
rz(0.6080724990587827) q[14];
ry(0.03225915052421118) q[15];
rz(2.085978365970856) q[15];
ry(0.0010327648574221264) q[16];
rz(-2.6106941643381005) q[16];
ry(-3.1372914676143586) q[17];
rz(-3.094350379032359) q[17];
ry(1.7052967062425306) q[18];
rz(0.04402936866355107) q[18];
ry(2.259860906355436) q[19];
rz(-0.3123157550041232) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.8761735754798863) q[0];
rz(-3.0112315023545353) q[0];
ry(-0.019229158020650058) q[1];
rz(-2.073309530396254) q[1];
ry(-3.141152432034088) q[2];
rz(-0.7331494874440452) q[2];
ry(-3.140585750861878) q[3];
rz(-2.3094584273754406) q[3];
ry(3.1259293239268393) q[4];
rz(-2.5221558341993515) q[4];
ry(-3.135453244209434) q[5];
rz(0.18927323587563932) q[5];
ry(-2.9181716140341836) q[6];
rz(0.7238375537989654) q[6];
ry(-0.020646669259002156) q[7];
rz(0.3158397844175802) q[7];
ry(0.4088128052476243) q[8];
rz(2.600670188601178) q[8];
ry(0.606665586876133) q[9];
rz(-1.002741096334596) q[9];
ry(0.6935933155211231) q[10];
rz(-3.0105960895661488) q[10];
ry(0.04409741432151914) q[11];
rz(-2.726380673866891) q[11];
ry(-0.36068441543815755) q[12];
rz(-2.1108196946432836) q[12];
ry(-0.72378734583135) q[13];
rz(-0.46656494485842614) q[13];
ry(-0.00941280815870904) q[14];
rz(-0.9577462263383834) q[14];
ry(-0.6132050010309644) q[15];
rz(-0.08298535337116969) q[15];
ry(0.9668192095315256) q[16];
rz(0.3675989923292198) q[16];
ry(3.062292686899235) q[17];
rz(1.46716174720988) q[17];
ry(-1.6326183057891575) q[18];
rz(1.9612491718409093) q[18];
ry(1.5168063718921605) q[19];
rz(-1.6319441895422944) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.4790316736357241) q[0];
rz(1.4727087186363894) q[0];
ry(-0.5799089969920992) q[1];
rz(1.5665580824740886) q[1];
ry(-0.8081828610562818) q[2];
rz(-2.3609085065819744) q[2];
ry(-0.18153421341063788) q[3];
rz(2.9634711658009762) q[3];
ry(2.5315528212084994) q[4];
rz(-0.07993834158263853) q[4];
ry(1.9079911108173047) q[5];
rz(-0.3635239827082631) q[5];
ry(0.30738673934916516) q[6];
rz(1.5973908525142109) q[6];
ry(0.7534034366844402) q[7];
rz(-0.249586015747365) q[7];
ry(-1.5813632327006828) q[8];
rz(1.862093634729725) q[8];
ry(1.2208615560751859) q[9];
rz(-1.9267661614479161) q[9];
ry(0.0648132266660264) q[10];
rz(0.4548446722123041) q[10];
ry(3.1373178464645233) q[11];
rz(2.4755586158417486) q[11];
ry(-0.07475994419643285) q[12];
rz(2.0267101567714283) q[12];
ry(-1.6689187465983917) q[13];
rz(-3.061745306223786) q[13];
ry(1.592659678586614) q[14];
rz(2.2777647435602546) q[14];
ry(2.331693773111597) q[15];
rz(-1.2008161510083315) q[15];
ry(0.012287571024593475) q[16];
rz(-2.1117881096311875) q[16];
ry(-0.2361294537757157) q[17];
rz(-2.5908973300593283) q[17];
ry(-0.5162141528474273) q[18];
rz(-0.48538383965380666) q[18];
ry(0.6469801595923146) q[19];
rz(-2.635908451872036) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.2894400423616521) q[0];
rz(2.686782684160371) q[0];
ry(-3.1056005647519953) q[1];
rz(-2.5164391763183165) q[1];
ry(0.0017738195497264823) q[2];
rz(-0.52867980753808) q[2];
ry(0.36332857503759364) q[3];
rz(-2.934794773700198) q[3];
ry(0.6997884903528645) q[4];
rz(-0.8314485127788718) q[4];
ry(-1.8478088383259736) q[5];
rz(-2.507940953740726) q[5];
ry(-0.06009935312516578) q[6];
rz(-1.0593246442128714) q[6];
ry(0.0053222575031526284) q[7];
rz(0.10666334835401778) q[7];
ry(3.1404830298235042) q[8];
rz(1.859105210341892) q[8];
ry(3.137668089399966) q[9];
rz(-1.9378373762879921) q[9];
ry(-0.4181221694104149) q[10];
rz(-2.0093303526638104) q[10];
ry(-3.1206237222256337) q[11];
rz(3.123772554783047) q[11];
ry(-2.3600638035474977) q[12];
rz(-2.210552282054695) q[12];
ry(3.103695360733478) q[13];
rz(0.34154954708599483) q[13];
ry(0.38188100177384326) q[14];
rz(0.9754192200516609) q[14];
ry(-0.27450719808469687) q[15];
rz(-0.08234098443491629) q[15];
ry(0.015847702667251547) q[16];
rz(-3.041921152718884) q[16];
ry(-0.2082840188051498) q[17];
rz(-2.312165068924251) q[17];
ry(1.3041754909563845) q[18];
rz(-0.8083645263970417) q[18];
ry(-0.8477763950167341) q[19];
rz(-1.8805241340153167) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.9752862675839736) q[0];
rz(1.953186470306304) q[0];
ry(-1.843603908584603) q[1];
rz(2.7822568314830125) q[1];
ry(-2.703581393288276) q[2];
rz(0.36471264835900524) q[2];
ry(3.130145898419978) q[3];
rz(0.0654514885607762) q[3];
ry(-0.018776433228262057) q[4];
rz(-0.41610157096926326) q[4];
ry(-0.76967896559644) q[5];
rz(1.5920679777716944) q[5];
ry(-0.2607415255004456) q[6];
rz(1.754103090750685) q[6];
ry(0.674988929732695) q[7];
rz(2.449102642439926) q[7];
ry(-1.5555097012871988) q[8];
rz(-0.716895070947952) q[8];
ry(1.0914817545662912) q[9];
rz(-1.436190359000518) q[9];
ry(-0.03200243996796548) q[10];
rz(2.508539612765726) q[10];
ry(-3.1372664697763173) q[11];
rz(1.2178430774026943) q[11];
ry(0.2663048772891753) q[12];
rz(-1.4562328031979799) q[12];
ry(-0.05934647537519085) q[13];
rz(1.2834106223123616) q[13];
ry(-1.6742369300862308) q[14];
rz(-0.8489239772934117) q[14];
ry(1.6192676876746452) q[15];
rz(-1.6709051474025354) q[15];
ry(-2.6881499358881498) q[16];
rz(-0.09274541236148348) q[16];
ry(-0.05770870109438153) q[17];
rz(0.4734737120277987) q[17];
ry(-2.9712520626671197) q[18];
rz(-0.7279702516649307) q[18];
ry(-2.9190306755347684) q[19];
rz(-0.8067292714089308) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.43398346295543533) q[0];
rz(0.7122879839660068) q[0];
ry(3.139187197240156) q[1];
rz(0.8311186536886274) q[1];
ry(-0.005620583562564896) q[2];
rz(-2.7941229597857933) q[2];
ry(-0.6792306099266759) q[3];
rz(-2.4695098157850124) q[3];
ry(2.103929313625388) q[4];
rz(-1.408002530923791) q[4];
ry(0.5923709232934913) q[5];
rz(-0.6193435607860573) q[5];
ry(-1.449074045123031) q[6];
rz(0.676743118725415) q[6];
ry(0.1232734459170901) q[7];
rz(0.7264006028730865) q[7];
ry(2.7578503422317135) q[8];
rz(1.5017235960492634) q[8];
ry(-2.9987301068673937) q[9];
rz(2.81101534158139) q[9];
ry(0.6617127758686958) q[10];
rz(-1.393132357616603) q[10];
ry(2.0848818928327724) q[11];
rz(-2.27122606969225) q[11];
ry(1.4042257579390696) q[12];
rz(1.1970105014031056) q[12];
ry(-2.83496549759795) q[13];
rz(1.6434261319491934) q[13];
ry(2.7362460146533087) q[14];
rz(-3.1203692687046267) q[14];
ry(1.8775379705343278) q[15];
rz(1.8943586006401496) q[15];
ry(-1.268946646838549) q[16];
rz(1.1981346465942453) q[16];
ry(-3.086209900917238) q[17];
rz(-1.9067281849287956) q[17];
ry(2.5679632357265803) q[18];
rz(1.4721105436636677) q[18];
ry(-0.8529358981709874) q[19];
rz(0.4265464844217712) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.9830523328800398) q[0];
rz(-1.344661400782873) q[0];
ry(2.1310625232383575) q[1];
rz(1.7014150972997053) q[1];
ry(-0.7327373539146933) q[2];
rz(-0.19665098943017764) q[2];
ry(3.119005793498292) q[3];
rz(-2.642772980800905) q[3];
ry(3.11272172425025) q[4];
rz(-1.4743037045780039) q[4];
ry(-2.9100168059970972) q[5];
rz(1.5475006040910437) q[5];
ry(3.1063924962697222) q[6];
rz(-2.5388117143804814) q[6];
ry(-0.0009187488870852077) q[7];
rz(-0.30164983274397505) q[7];
ry(3.1255944781909086) q[8];
rz(-0.30673126667507655) q[8];
ry(-3.0812932539623072) q[9];
rz(0.05236880179506987) q[9];
ry(-2.073219699983256) q[10];
rz(0.46217983076433633) q[10];
ry(-2.6584749804379895) q[11];
rz(-2.906257474255851) q[11];
ry(0.5569065069722132) q[12];
rz(-0.49848917632566003) q[12];
ry(-0.07492781219329192) q[13];
rz(2.6751436998949774) q[13];
ry(0.002714310771249728) q[14];
rz(-2.265459849296432) q[14];
ry(0.011423631812895074) q[15];
rz(-1.9194299849515186) q[15];
ry(3.093630210363464) q[16];
rz(0.2744354807169529) q[16];
ry(2.5890641347246794) q[17];
rz(-0.4255750166661203) q[17];
ry(-1.0241731643181873) q[18];
rz(1.0660780523924682) q[18];
ry(1.737498604541527) q[19];
rz(0.030145721955679328) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.53158887002989) q[0];
rz(-0.8541880631720993) q[0];
ry(1.6868955501535288) q[1];
rz(0.6294652898209778) q[1];
ry(3.1218242681427424) q[2];
rz(2.808940910532435) q[2];
ry(2.7593077023575563) q[3];
rz(2.8297429565744325) q[3];
ry(1.2100145212339397) q[4];
rz(3.13377208805435) q[4];
ry(-2.355729962431642) q[5];
rz(-1.7330733529565014) q[5];
ry(-2.025356770919198) q[6];
rz(-0.44738274419374857) q[6];
ry(-1.9936521205162825) q[7];
rz(1.323355473696674) q[7];
ry(-0.5863283586304736) q[8];
rz(2.3698935505521312) q[8];
ry(-3.1361683266072147) q[9];
rz(3.013278545487552) q[9];
ry(0.014314831316195476) q[10];
rz(-0.8704603459837932) q[10];
ry(0.0002445375931543481) q[11];
rz(2.9027642375136042) q[11];
ry(-0.017152286301062455) q[12];
rz(0.3286929356830585) q[12];
ry(-3.1274293953815087) q[13];
rz(-3.12544798122027) q[13];
ry(-1.3946885486897118) q[14];
rz(1.1886423909349797) q[14];
ry(1.846254272282902) q[15];
rz(-2.6841713134349114) q[15];
ry(0.1613546955798144) q[16];
rz(0.39696378202403176) q[16];
ry(-2.9882603440312696) q[17];
rz(2.15275476747179) q[17];
ry(2.8525007047572744) q[18];
rz(-2.8820410753287784) q[18];
ry(-2.554691201069648) q[19];
rz(1.2857055412434786) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.4373313053681285) q[0];
rz(-1.6298252468404297) q[0];
ry(0.04221700840002285) q[1];
rz(1.5676187673229836) q[1];
ry(3.1256946556538714) q[2];
rz(2.0032279505992028) q[2];
ry(1.4775565229356151) q[3];
rz(0.9645652290437692) q[3];
ry(-3.1182076675901835) q[4];
rz(-1.3677906898389725) q[4];
ry(2.2808796156706146) q[5];
rz(-1.0703075689286434) q[5];
ry(0.07208884279003486) q[6];
rz(0.8488809594920559) q[6];
ry(-0.004847451292787852) q[7];
rz(-0.890142431825339) q[7];
ry(0.005328215355141275) q[8];
rz(-1.0098420930499614) q[8];
ry(-3.107042568387664) q[9];
rz(0.7739431067775925) q[9];
ry(1.3564359315610117) q[10];
rz(-0.8595037813463051) q[10];
ry(-0.49382274573666873) q[11];
rz(0.9774783089781496) q[11];
ry(0.7127120286419744) q[12];
rz(-1.2410389204014909) q[12];
ry(-3.1127533437211996) q[13];
rz(-1.4205544351998016) q[13];
ry(0.5529525328064366) q[14];
rz(3.1334151402320534) q[14];
ry(-0.03204758673688736) q[15];
rz(0.5376830201196857) q[15];
ry(-2.8953080721911735) q[16];
rz(1.6378586870142364) q[16];
ry(-2.74958610271893) q[17];
rz(-2.241957525865742) q[17];
ry(-1.3287822894309063) q[18];
rz(2.3507884984946426) q[18];
ry(-0.9294569509742949) q[19];
rz(2.375285971649846) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.6056257048283902) q[0];
rz(2.6034020985060646) q[0];
ry(-0.1258164411957595) q[1];
rz(1.566266815393135) q[1];
ry(-1.55663176452099) q[2];
rz(-3.1325410250149015) q[2];
ry(-0.14245691161637117) q[3];
rz(2.6687795336651066) q[3];
ry(-0.32078600027734705) q[4];
rz(2.192997714049568) q[4];
ry(-1.4646056751496612) q[5];
rz(-2.2479392388137516) q[5];
ry(-1.4114052409190085) q[6];
rz(-2.407698819864308) q[6];
ry(1.8420827057923437) q[7];
rz(2.9475027420092945) q[7];
ry(-2.4641301188851643) q[8];
rz(0.9235957555663507) q[8];
ry(-0.05808842414213089) q[9];
rz(1.831488204933277) q[9];
ry(-1.1303384728753174) q[10];
rz(-0.010562818931797828) q[10];
ry(0.016742368225410354) q[11];
rz(-0.7288897853565962) q[11];
ry(-3.102828447460013) q[12];
rz(2.5048795265655) q[12];
ry(-0.02056909415470809) q[13];
rz(1.946661221596088) q[13];
ry(-1.367144876848121) q[14];
rz(-0.14204197018272602) q[14];
ry(0.012983651661537365) q[15];
rz(-1.658060994730492) q[15];
ry(3.0565564321344745) q[16];
rz(1.9795108046738719) q[16];
ry(-2.9798459927429146) q[17];
rz(0.7175901046988805) q[17];
ry(-0.49364916812303783) q[18];
rz(-1.3767087861346035) q[18];
ry(1.6134792368613184) q[19];
rz(1.8173002895530168) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.6686250667910625) q[0];
rz(1.8979439864316143) q[0];
ry(1.5691762663391482) q[1];
rz(-0.43573778123770346) q[1];
ry(-3.1035419028723132) q[2];
rz(-0.0024127019958631686) q[2];
ry(-0.013086770242738532) q[3];
rz(-2.198729579363426) q[3];
ry(-3.1286094983407713) q[4];
rz(-2.8736650208220724) q[4];
ry(0.12340294017144988) q[5];
rz(3.1043939888296372) q[5];
ry(-0.03056912457945149) q[6];
rz(2.5334194225342794) q[6];
ry(-0.031379637411629524) q[7];
rz(1.827431959260048) q[7];
ry(3.1067338068850443) q[8];
rz(-0.2631379429661367) q[8];
ry(0.011432449632005844) q[9];
rz(-1.7118060977188443) q[9];
ry(-0.4737945409721589) q[10];
rz(2.5743544303722405) q[10];
ry(-3.1254503076471174) q[11];
rz(2.597307484254245) q[11];
ry(0.13819980810418558) q[12];
rz(2.7346970615727497) q[12];
ry(0.038927490742165276) q[13];
rz(1.4641271415689232) q[13];
ry(-2.5331334039815827) q[14];
rz(-1.1074061963721693) q[14];
ry(3.137854559317568) q[15];
rz(-1.6993667751987793) q[15];
ry(0.4073851796006973) q[16];
rz(0.5827821324803203) q[16];
ry(-2.9439818695906363) q[17];
rz(1.0679668065095622) q[17];
ry(-3.111856294063385) q[18];
rz(-1.4168124098313744) q[18];
ry(-1.490574929926465) q[19];
rz(2.4965363203976687) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.568977150232226) q[0];
rz(-3.140504955418756) q[0];
ry(-0.009039719393849845) q[1];
rz(-1.134296750766185) q[1];
ry(0.28958367882305963) q[2];
rz(-1.5559222056221507) q[2];
ry(-3.078094028502164) q[3];
rz(-0.21358359731651447) q[3];
ry(-0.15500664434757194) q[4];
rz(3.0160239519369774) q[4];
ry(-0.17057741067977972) q[5];
rz(-0.7265814357775522) q[5];
ry(-0.29562210268597167) q[6];
rz(-3.091108703690357) q[6];
ry(-0.1063843678961168) q[7];
rz(-0.5506851486662824) q[7];
ry(2.451935776818602) q[8];
rz(1.2032460471025332) q[8];
ry(-1.8877574826598202) q[9];
rz(2.0526539334879486) q[9];
ry(0.18217399019291225) q[10];
rz(2.0109842785026517) q[10];
ry(2.0124618565634713) q[11];
rz(1.5859916121454494) q[11];
ry(2.1052184962056613) q[12];
rz(-1.731491466797519) q[12];
ry(-0.2636264190162336) q[13];
rz(-1.0630989880128052) q[13];
ry(-3.0641529885097243) q[14];
rz(0.4810511816302085) q[14];
ry(3.030929025133824) q[15];
rz(0.736898104675836) q[15];
ry(2.8821034405935584) q[16];
rz(0.3462178387995108) q[16];
ry(1.8359886422970804) q[17];
rz(0.03748241619917092) q[17];
ry(-1.2495369187742922) q[18];
rz(-0.3171607262503053) q[18];
ry(0.1472524881189079) q[19];
rz(0.27381738895823116) q[19];