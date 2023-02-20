OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.241918573189694) q[0];
rz(-0.650633338030363) q[0];
ry(-0.9751103053012319) q[1];
rz(2.6563589611324003) q[1];
ry(0.43877263209911455) q[2];
rz(2.71890201719667) q[2];
ry(-0.014863617544532725) q[3];
rz(-2.241758127772323) q[3];
ry(1.482750634353535) q[4];
rz(0.5316908817809853) q[4];
ry(0.5304661479499161) q[5];
rz(-1.9615729106797142) q[5];
ry(0.1170060578416301) q[6];
rz(-0.3068862186815409) q[6];
ry(-2.678271287509742) q[7];
rz(1.9016970042596375) q[7];
ry(2.9768137347492893) q[8];
rz(2.257753072432651) q[8];
ry(-3.067520202213582) q[9];
rz(0.1842158229740072) q[9];
ry(1.1299164629232559) q[10];
rz(-0.36077152133405754) q[10];
ry(0.34460176777071816) q[11];
rz(1.563183697375326) q[11];
ry(-3.089865775752957) q[12];
rz(1.1702434403722632) q[12];
ry(0.17606429053762174) q[13];
rz(0.6018340372449265) q[13];
ry(2.339398385416934) q[14];
rz(-0.7579355702703592) q[14];
ry(0.36213118585908344) q[15];
rz(-1.4835193065555994) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.702272529690563) q[0];
rz(-1.8439137870454319) q[0];
ry(-2.9615859447399364) q[1];
rz(0.6832841048967254) q[1];
ry(-0.22647840320690327) q[2];
rz(2.299728545890028) q[2];
ry(0.04479959453267579) q[3];
rz(1.201984195947154) q[3];
ry(2.4184542375111886) q[4];
rz(1.2737595205691736) q[4];
ry(-0.593237297478935) q[5];
rz(-2.5801340663894403) q[5];
ry(0.7260813916687647) q[6];
rz(-2.049686973102838) q[6];
ry(-1.4978615449571917) q[7];
rz(-2.625645262636086) q[7];
ry(-1.7219610431586219) q[8];
rz(-0.15794125381527027) q[8];
ry(0.7638559603657296) q[9];
rz(-0.9525906090790528) q[9];
ry(3.0757653185478513) q[10];
rz(1.1177200352495262) q[10];
ry(-0.08292847419153393) q[11];
rz(-2.6633157382528334) q[11];
ry(3.132352179218338) q[12];
rz(-2.6907028748108672) q[12];
ry(0.015270945398842883) q[13];
rz(-2.2620804581173246) q[13];
ry(-2.1885034298321955) q[14];
rz(-1.535219540154201) q[14];
ry(0.7463129590182163) q[15];
rz(1.9750545236447636) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.6113976597933806) q[0];
rz(-1.6454477228620439) q[0];
ry(2.61862421853542) q[1];
rz(-0.5090972623191496) q[1];
ry(-0.3436586941825092) q[2];
rz(-0.1830853089231574) q[2];
ry(2.626740337449476) q[3];
rz(-0.9321208639353209) q[3];
ry(-2.2603483374431437) q[4];
rz(2.819135538984451) q[4];
ry(2.8601398743075337) q[5];
rz(-2.501494749984323) q[5];
ry(-3.135271278978615) q[6];
rz(1.5103616453038606) q[6];
ry(-2.8565650106223224) q[7];
rz(0.23055112868549982) q[7];
ry(-1.072043495447965) q[8];
rz(-2.3831206403947833) q[8];
ry(-3.106427058786345) q[9];
rz(2.9023943673786095) q[9];
ry(2.3715664547085504) q[10];
rz(-2.0744994853660277) q[10];
ry(0.29036501799311054) q[11];
rz(0.2003375442341348) q[11];
ry(-2.417655173422869) q[12];
rz(0.09278771991429927) q[12];
ry(1.2483718963379413) q[13];
rz(-0.960483634925738) q[13];
ry(2.9590844499628797) q[14];
rz(0.857613730142198) q[14];
ry(0.20011874461771928) q[15];
rz(-0.079065836228945) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.990831425250611) q[0];
rz(0.44463767675713145) q[0];
ry(-1.6991191606625247) q[1];
rz(2.887026714696898) q[1];
ry(-0.02961773899262943) q[2];
rz(-3.091137981958968) q[2];
ry(-0.026841148525741158) q[3];
rz(1.0279515384671745) q[3];
ry(0.06769520100357784) q[4];
rz(-2.64554044380533) q[4];
ry(-1.6308531431809277) q[5];
rz(2.545344430184437) q[5];
ry(2.3668978004235957) q[6];
rz(0.05605360799270381) q[6];
ry(-1.6290032914620465) q[7];
rz(2.008583777553644) q[7];
ry(1.927853650272258) q[8];
rz(-2.004168515364473) q[8];
ry(-0.4332210534562098) q[9];
rz(0.6700169944707098) q[9];
ry(2.8288457640442752) q[10];
rz(2.1416197509415253) q[10];
ry(0.009923869086903006) q[11];
rz(-2.2221387651637263) q[11];
ry(-1.2783558921436387) q[12];
rz(-3.137544391186783) q[12];
ry(3.1220520352337333) q[13];
rz(-2.4053214018672486) q[13];
ry(0.006366002777542512) q[14];
rz(2.5838522284346053) q[14];
ry(2.0173961629905355) q[15];
rz(0.6129060099686566) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.012154206684908964) q[0];
rz(1.2846279582278737) q[0];
ry(-1.3156785771654897) q[1];
rz(-0.474376410484437) q[1];
ry(0.0076441917578922115) q[2];
rz(0.4409587056407113) q[2];
ry(0.5313585117409095) q[3];
rz(-2.527508168091902) q[3];
ry(-2.6463160119377163) q[4];
rz(1.4321195995619018) q[4];
ry(0.15544951133551282) q[5];
rz(0.7637000635810752) q[5];
ry(-0.018082663874994676) q[6];
rz(1.468201624709715) q[6];
ry(-2.65354198326662) q[7];
rz(-0.01058038998686401) q[7];
ry(3.1338419810111513) q[8];
rz(1.5798630551329964) q[8];
ry(-3.039929338100368) q[9];
rz(-0.3616890976668036) q[9];
ry(1.7957150858702962) q[10];
rz(2.8737592679512067) q[10];
ry(-1.8708340525676703) q[11];
rz(3.0037030325877088) q[11];
ry(2.4689566773851284) q[12];
rz(2.2101066673807965) q[12];
ry(0.022748280126966947) q[13];
rz(-0.7560869553333195) q[13];
ry(-3.028265606848442) q[14];
rz(-1.8969784446493945) q[14];
ry(1.5604202585434221) q[15];
rz(-1.6714980071162344) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.4261024978201726) q[0];
rz(0.8870874220952717) q[0];
ry(-1.427437063977079) q[1];
rz(-2.5825941159788752) q[1];
ry(0.11908365764030915) q[2];
rz(3.0455087704035826) q[2];
ry(0.9724910107333766) q[3];
rz(-0.5565628590363119) q[3];
ry(2.149577139519395) q[4];
rz(-0.6845214343297162) q[4];
ry(2.7435779202120885) q[5];
rz(-2.779522061923047) q[5];
ry(0.6206080799556339) q[6];
rz(-1.678712463972621) q[6];
ry(1.281465267420526) q[7];
rz(-0.6647624026291531) q[7];
ry(2.2881416672300303) q[8];
rz(2.7203818073995074) q[8];
ry(3.0603359877466487) q[9];
rz(2.6023749013612827) q[9];
ry(-0.08500899525721017) q[10];
rz(-0.380147619148411) q[10];
ry(1.8709410991043716) q[11];
rz(-2.7012239838056007) q[11];
ry(-1.4000842105951259) q[12];
rz(-0.5421248397945527) q[12];
ry(-1.861936606465859) q[13];
rz(1.8888864605402729) q[13];
ry(3.078211641798181) q[14];
rz(-2.3685731210182106) q[14];
ry(-2.302892247857343) q[15];
rz(-2.957910932049086) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.2455375567464166) q[0];
rz(0.5508637687506086) q[0];
ry(-0.8935435133157841) q[1];
rz(1.3916555127970693) q[1];
ry(0.3090117626452491) q[2];
rz(-1.775998212433799) q[2];
ry(-3.139837919700325) q[3];
rz(-2.8077960768406465) q[3];
ry(-3.067839342898511) q[4];
rz(-2.729333866891464) q[4];
ry(-0.3305423564699093) q[5];
rz(3.0447075772371) q[5];
ry(-1.3846437205694493) q[6];
rz(-0.6217356036427906) q[6];
ry(1.1470885925769991) q[7];
rz(-0.32924424042777445) q[7];
ry(0.9730736776129794) q[8];
rz(-2.5741581044510324) q[8];
ry(-3.1291937545982766) q[9];
rz(-3.081114057012232) q[9];
ry(0.297720918985954) q[10];
rz(-2.4313630339510732) q[10];
ry(0.0030991715268475772) q[11];
rz(2.5583537211067418) q[11];
ry(0.003970607474684584) q[12];
rz(-1.5626283567069024) q[12];
ry(3.114513323731901) q[13];
rz(0.37700307272878414) q[13];
ry(-1.9330328856319932) q[14];
rz(0.47199521365792396) q[14];
ry(2.726014587156324) q[15];
rz(0.5218820987970093) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.9888589352690583) q[0];
rz(2.331403570732047) q[0];
ry(0.5668886430855951) q[1];
rz(-2.6588356828468793) q[1];
ry(1.5625159681890404) q[2];
rz(-0.678688492602517) q[2];
ry(2.270176836941898) q[3];
rz(0.6725435520457747) q[3];
ry(2.332471856488942) q[4];
rz(0.31960657631754497) q[4];
ry(-3.064441277902712) q[5];
rz(-1.2122099213184725) q[5];
ry(3.042891900491039) q[6];
rz(-0.5998342352485784) q[6];
ry(-3.081758976119017) q[7];
rz(2.5780187923982645) q[7];
ry(-2.0051147139005803) q[8];
rz(1.713319339592187) q[8];
ry(0.06602828080627354) q[9];
rz(2.0913234139469266) q[9];
ry(-3.0521423616763714) q[10];
rz(-1.6606830813992286) q[10];
ry(-1.0990309289738929) q[11];
rz(-1.3874784403868272) q[11];
ry(-1.2178303273068067) q[12];
rz(-2.4850667203474557) q[12];
ry(-1.4635147495956111) q[13];
rz(-0.5987467758672427) q[13];
ry(3.0428065787915584) q[14];
rz(0.5362023116503138) q[14];
ry(-3.1040738449941574) q[15];
rz(-0.9329594937579664) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.708336907095941) q[0];
rz(2.751160288624332) q[0];
ry(-1.67156408711561) q[1];
rz(0.28404786310296704) q[1];
ry(-0.5015294269838062) q[2];
rz(2.058659863255701) q[2];
ry(3.095644000970622) q[3];
rz(-1.0668741220360767) q[3];
ry(-0.30841696217286185) q[4];
rz(-3.062955205111372) q[4];
ry(-0.16624573873055226) q[5];
rz(-2.264054327156892) q[5];
ry(1.3230243675847753) q[6];
rz(2.833335453852835) q[6];
ry(1.3170917243196811) q[7];
rz(-0.3536782198989216) q[7];
ry(-0.47056604844529915) q[8];
rz(1.1559099705957367) q[8];
ry(-0.06309916580064368) q[9];
rz(-0.4986419120357581) q[9];
ry(-2.6918574533792774) q[10];
rz(-1.7532277589523715) q[10];
ry(-0.5420623350009013) q[11];
rz(0.34404791740203183) q[11];
ry(2.8563201642910694) q[12];
rz(0.6012447390634342) q[12];
ry(0.30424657866118654) q[13];
rz(-2.2437508218743947) q[13];
ry(-0.9701110932851034) q[14];
rz(-0.43741593103651966) q[14];
ry(2.20795893960161) q[15];
rz(0.5598343185368037) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.633844576222294) q[0];
rz(-2.999463892332631) q[0];
ry(1.6699600056907058) q[1];
rz(-2.363359476747343) q[1];
ry(-0.2191718650263086) q[2];
rz(3.0930898899892587) q[2];
ry(-2.4527945482817586) q[3];
rz(-0.0727065878477424) q[3];
ry(-2.638867999868051) q[4];
rz(-2.011989895455776) q[4];
ry(-2.7838205583734625) q[5];
rz(0.14179106119267804) q[5];
ry(-1.4033148299755673) q[6];
rz(-1.3875505101023797) q[6];
ry(2.791278470777649) q[7];
rz(-0.35105129311660976) q[7];
ry(2.150632287683927) q[8];
rz(1.032633883625783) q[8];
ry(0.4320896201710651) q[9];
rz(-2.3668543592675038) q[9];
ry(-3.1006217514510563) q[10];
rz(0.650541379524067) q[10];
ry(-3.1229889733332024) q[11];
rz(2.0917364240721095) q[11];
ry(3.014640212768778) q[12];
rz(0.5580462909603909) q[12];
ry(0.5726470639402557) q[13];
rz(-1.883035045233677) q[13];
ry(-0.13624018455018927) q[14];
rz(-1.877502047765943) q[14];
ry(-1.8923111130990566) q[15];
rz(2.816230068015008) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.8579255532259538) q[0];
rz(-1.1261860559801047) q[0];
ry(0.7320498735228936) q[1];
rz(1.168288201189715) q[1];
ry(-0.975570525822127) q[2];
rz(1.530973708898035) q[2];
ry(0.08412470287180085) q[3];
rz(-0.4952540785108015) q[3];
ry(3.075890489613992) q[4];
rz(0.8118548704428052) q[4];
ry(2.298519000307552) q[5];
rz(-1.0663033091748013) q[5];
ry(3.088785997975264) q[6];
rz(-2.43295596721074) q[6];
ry(1.3522821879738265) q[7];
rz(3.0742832753195817) q[7];
ry(-2.7159495528772934) q[8];
rz(-2.2476556456768657) q[8];
ry(0.5748238126379402) q[9];
rz(-2.451516453531062) q[9];
ry(-0.9773458209740639) q[10];
rz(-0.02941150811854065) q[10];
ry(1.4735883383294448) q[11];
rz(0.06203628409669548) q[11];
ry(-2.3625598680799635) q[12];
rz(1.3263635468511343) q[12];
ry(2.7558325228197513) q[13];
rz(0.36815215114562677) q[13];
ry(-0.030365965905447323) q[14];
rz(0.8106949919734309) q[14];
ry(-0.9155864138479001) q[15];
rz(-2.976268864791196) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.2739614886267312) q[0];
rz(-1.9514888888209363) q[0];
ry(-2.7260933560681657) q[1];
rz(-3.073495970063305) q[1];
ry(-3.1313055603496585) q[2];
rz(1.6213090807703194) q[2];
ry(3.1011749867107152) q[3];
rz(-0.35425410614524033) q[3];
ry(-2.9835028883740127) q[4];
rz(2.9216299741280434) q[4];
ry(-0.1591227194503508) q[5];
rz(-1.7315798151079698) q[5];
ry(0.2780920974089489) q[6];
rz(0.983466748252008) q[6];
ry(1.8865982700871653) q[7];
rz(-1.082792231318052) q[7];
ry(-0.05794051888017737) q[8];
rz(0.9554739927666881) q[8];
ry(0.3696408169713516) q[9];
rz(2.9574503099847536) q[9];
ry(0.10963325719711037) q[10];
rz(-0.11802585427268307) q[10];
ry(0.12013169559933964) q[11];
rz(-2.8860336675252434) q[11];
ry(-2.897985328166513) q[12];
rz(-3.0925170450700943) q[12];
ry(-3.009811650306921) q[13];
rz(-1.0477191209350956) q[13];
ry(-3.0848522908584775) q[14];
rz(-2.0018992428799534) q[14];
ry(1.6572709964972083) q[15];
rz(0.9297978557130043) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.5816917615477792) q[0];
rz(2.8927341755735965) q[0];
ry(0.6352000729214583) q[1];
rz(-0.9169260255804932) q[1];
ry(0.6227494471169884) q[2];
rz(2.5548789094586355) q[2];
ry(2.9697774948046476) q[3];
rz(2.089269578095) q[3];
ry(-1.3232129407890325) q[4];
rz(-3.0894355867618897) q[4];
ry(-0.992534481557314) q[5];
rz(-0.4299624717205371) q[5];
ry(-3.004353925767534) q[6];
rz(-2.851271208576318) q[6];
ry(2.9744309521580146) q[7];
rz(-2.5145244538807616) q[7];
ry(0.3622090244540266) q[8];
rz(-1.5138466677827231) q[8];
ry(-0.9783474931754518) q[9];
rz(-1.6902715969491402) q[9];
ry(-0.01799306043097129) q[10];
rz(1.152642653343921) q[10];
ry(0.9354446054438723) q[11];
rz(-1.1256006819072821) q[11];
ry(-0.2759090700179595) q[12];
rz(3.0696028240386664) q[12];
ry(-1.2549155704234503) q[13];
rz(0.9371000906893023) q[13];
ry(-1.7005914360977963) q[14];
rz(2.7067886056310297) q[14];
ry(1.2137097404007582) q[15];
rz(-3.12144949493251) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.9478060928934182) q[0];
rz(0.63239383236309) q[0];
ry(-0.8241215892527767) q[1];
rz(2.4290567366351286) q[1];
ry(-0.38851454490515724) q[2];
rz(-2.0890078145434146) q[2];
ry(-3.1160718378644754) q[3];
rz(-1.0001327299693576) q[3];
ry(2.219785907920314) q[4];
rz(-0.010000936735448107) q[4];
ry(1.4736063038547542) q[5];
rz(0.058480844051958414) q[5];
ry(-0.18279351500419552) q[6];
rz(-1.7870337443700017) q[6];
ry(-1.3524660046614416) q[7];
rz(0.09204624008426728) q[7];
ry(3.0617874548029387) q[8];
rz(2.9244378510761324) q[8];
ry(-3.1024178660930017) q[9];
rz(-2.337459135319296) q[9];
ry(-0.5227339030226866) q[10];
rz(2.553835263994459) q[10];
ry(0.4392656700848977) q[11];
rz(-2.4996902974707) q[11];
ry(0.03268282918849424) q[12];
rz(-1.7549019457941393) q[12];
ry(-2.9372753246019414) q[13];
rz(1.7134920600706254) q[13];
ry(-2.2666926487113903) q[14];
rz(-1.3006243290239006) q[14];
ry(3.1025503832975936) q[15];
rz(1.4277560007382233) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.658960895773584) q[0];
rz(-1.2554309377961392) q[0];
ry(-0.5406990362543638) q[1];
rz(-1.2288351966460587) q[1];
ry(-3.0265471400599155) q[2];
rz(2.1196083951471563) q[2];
ry(-0.44837062555338575) q[3];
rz(2.884011910967892) q[3];
ry(-2.8957675093672113) q[4];
rz(-2.0566294261936786) q[4];
ry(2.5183451825483174) q[5];
rz(2.957923597009016) q[5];
ry(1.5649071950816467) q[6];
rz(-2.7235069728877725) q[6];
ry(2.0418601973651156) q[7];
rz(-1.3612057483657203) q[7];
ry(2.74100980997503) q[8];
rz(2.6222157111627196) q[8];
ry(0.36246756874134145) q[9];
rz(-3.071545949819054) q[9];
ry(0.4220565960215489) q[10];
rz(-2.6637300578909926) q[10];
ry(-1.1727110734931327) q[11];
rz(2.695803322532756) q[11];
ry(-1.7676264132815096) q[12];
rz(-1.786779267751042) q[12];
ry(2.217512660031663) q[13];
rz(-2.10297769181601) q[13];
ry(0.0028943962640086696) q[14];
rz(2.406054977096033) q[14];
ry(-0.49275347491226995) q[15];
rz(0.880571878531951) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.27781864471208) q[0];
rz(-2.56196902063323) q[0];
ry(1.28555999858526) q[1];
rz(2.479222590893649) q[1];
ry(-2.9027406557905175) q[2];
rz(0.016269258500819284) q[2];
ry(0.22374538747339823) q[3];
rz(1.8475795175481187) q[3];
ry(0.05073230412427564) q[4];
rz(1.5279088623776014) q[4];
ry(0.21757262633421523) q[5];
rz(2.4006968184644912) q[5];
ry(-0.08876429068373604) q[6];
rz(0.7216584437899386) q[6];
ry(1.6071843332161873) q[7];
rz(-0.32772786431435824) q[7];
ry(0.39580190646952307) q[8];
rz(-2.7942101792406215) q[8];
ry(-1.7226306756182093) q[9];
rz(-0.08311828084338568) q[9];
ry(0.6342242825043218) q[10];
rz(-1.737860827296691) q[10];
ry(-0.004557275918442205) q[11];
rz(0.6524487337180392) q[11];
ry(3.1094372847066523) q[12];
rz(-1.3265419440427229) q[12];
ry(0.10895132797769413) q[13];
rz(-2.3053606635735684) q[13];
ry(-0.6683648208966515) q[14];
rz(-1.0364540803569975) q[14];
ry(-2.8012978771128694) q[15];
rz(-2.3525205800179303) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.9307499121005199) q[0];
rz(-2.849002951027631) q[0];
ry(2.510931953103596) q[1];
rz(1.00335058853541) q[1];
ry(-1.6007279240732046) q[2];
rz(-1.934870253144185) q[2];
ry(-0.003783268311349744) q[3];
rz(-1.2536859607540887) q[3];
ry(-0.917235305701829) q[4];
rz(1.1615484295722887) q[4];
ry(-0.02332606936822046) q[5];
rz(-0.2333773188160118) q[5];
ry(1.8679406888307517) q[6];
rz(0.05952996410619423) q[6];
ry(-0.3804442608644056) q[7];
rz(2.7012380034185717) q[7];
ry(-1.283655763697781) q[8];
rz(-2.8368455586123678) q[8];
ry(-0.2035273433185596) q[9];
rz(0.051054022240455986) q[9];
ry(-0.2150616679002805) q[10];
rz(-0.681886524549091) q[10];
ry(0.8428938418952531) q[11];
rz(2.6262383919667744) q[11];
ry(1.8421950763224826) q[12];
rz(-2.3396999450323546) q[12];
ry(-1.8133336820588593) q[13];
rz(1.6050304246365625) q[13];
ry(-0.1877190297009168) q[14];
rz(2.3088087306375815) q[14];
ry(-1.3938953665418883) q[15];
rz(2.1260268624257694) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.4326425540738241) q[0];
rz(-0.48794890102567123) q[0];
ry(-0.9332236076432077) q[1];
rz(-3.063789208599553) q[1];
ry(-1.0326348413593314) q[2];
rz(-1.653768476155566) q[2];
ry(-3.1087312990161005) q[3];
rz(2.7493402747185085) q[3];
ry(0.026235552076179447) q[4];
rz(3.0798432639614157) q[4];
ry(1.940053977502303) q[5];
rz(0.572783861888044) q[5];
ry(-3.103802229308863) q[6];
rz(0.4407053081861463) q[6];
ry(3.098444296785708) q[7];
rz(-2.4846573281202518) q[7];
ry(-0.14673983050758466) q[8];
rz(-0.32008123380429687) q[8];
ry(2.277319468677553) q[9];
rz(1.0593332029483644) q[9];
ry(1.9052051003546016) q[10];
rz(2.9121306958605095) q[10];
ry(3.1072502901227876) q[11];
rz(2.789206173130895) q[11];
ry(3.135960232028347) q[12];
rz(2.1925460233086485) q[12];
ry(-2.497973261875858) q[13];
rz(-0.35475661553235166) q[13];
ry(-2.7974300922322937) q[14];
rz(2.853447634621825) q[14];
ry(2.750386123139968) q[15];
rz(2.142869497829343) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.24612672613245312) q[0];
rz(0.7604693395337261) q[0];
ry(1.5300867239395357) q[1];
rz(0.46260811296680165) q[1];
ry(1.0969784078960645) q[2];
rz(2.100441529034196) q[2];
ry(2.7644407493221665) q[3];
rz(-1.4216897867467102) q[3];
ry(0.11188526995467574) q[4];
rz(-1.1511258030631213) q[4];
ry(-3.1197066912778526) q[5];
rz(0.5492371704081727) q[5];
ry(-0.07317427154506895) q[6];
rz(-2.0329573311683236) q[6];
ry(-0.5096489618197938) q[7];
rz(-0.006369650255060495) q[7];
ry(0.4229159844208985) q[8];
rz(0.07318278940662105) q[8];
ry(-3.135943865713517) q[9];
rz(-2.093459596963665) q[9];
ry(-0.3384990304278342) q[10];
rz(-2.4695571576423663) q[10];
ry(-0.6355575701724866) q[11];
rz(2.140319723663842) q[11];
ry(-2.580983847649032) q[12];
rz(-2.4713374056656727) q[12];
ry(-1.963439164819714) q[13];
rz(2.4473165448432104) q[13];
ry(-2.373721474740814) q[14];
rz(-1.1052318094921498) q[14];
ry(-1.0046384254054859) q[15];
rz(-2.735812979682781) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.1143427475305074) q[0];
rz(-2.126557121029286) q[0];
ry(-0.9866417083713152) q[1];
rz(0.6292458054970957) q[1];
ry(-1.0953999733781492) q[2];
rz(1.9129686059230546) q[2];
ry(-0.10534845022995203) q[3];
rz(-1.6477017726821621) q[3];
ry(0.026563521116866027) q[4];
rz(-2.388373785356367) q[4];
ry(-1.8564193648054959) q[5];
rz(1.6211299127313403) q[5];
ry(3.133825522062765) q[6];
rz(2.993581958527208) q[6];
ry(3.028263609837712) q[7];
rz(2.750390781141263) q[7];
ry(-1.2102312633885204) q[8];
rz(-0.8581648436993677) q[8];
ry(0.7919601325348005) q[9];
rz(0.01675091361496328) q[9];
ry(-1.075584886752857) q[10];
rz(1.8468316247600725) q[10];
ry(3.072970437125714) q[11];
rz(-2.0646943482808227) q[11];
ry(3.114526735733929) q[12];
rz(-2.6855125560556328) q[12];
ry(-2.211004634769541) q[13];
rz(-2.413086734667718) q[13];
ry(-0.43157107157197844) q[14];
rz(2.2305923964138454) q[14];
ry(2.8797686438919596) q[15];
rz(-2.7744570740715893) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.5523746305996795) q[0];
rz(-0.4662539204574395) q[0];
ry(2.7352220009379153) q[1];
rz(-3.1256103308141325) q[1];
ry(-2.4704949041459487) q[2];
rz(-0.509517493108806) q[2];
ry(-1.3647352309191092) q[3];
rz(2.735388816902335) q[3];
ry(0.6694246237428043) q[4];
rz(-0.5589050888407933) q[4];
ry(-2.8364687184680135) q[5];
rz(-0.12694509149504984) q[5];
ry(1.4317583756936694) q[6];
rz(1.7512146392169514) q[6];
ry(0.7960467186895107) q[7];
rz(-1.1271984354889337) q[7];
ry(-0.05220656748211963) q[8];
rz(0.6269081804268587) q[8];
ry(-0.12077819489050867) q[9];
rz(-0.026957560999578675) q[9];
ry(-1.7058680489765574) q[10];
rz(0.8620312918273414) q[10];
ry(0.07122186494763247) q[11];
rz(-2.160970601759331) q[11];
ry(0.6467190506154961) q[12];
rz(2.5800872091692493) q[12];
ry(1.5010916176913693) q[13];
rz(-0.918704920911716) q[13];
ry(2.717807655242883) q[14];
rz(0.9392382536044239) q[14];
ry(-2.407758975230397) q[15];
rz(-1.311089539890081) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.842065917378914) q[0];
rz(-1.9767438029830264) q[0];
ry(2.1280405178272357) q[1];
rz(2.7416017339800445) q[1];
ry(-1.8467520906593426) q[2];
rz(0.5292242014674979) q[2];
ry(0.24741519323593317) q[3];
rz(-1.876925250165205) q[3];
ry(3.138529560874542) q[4];
rz(-2.821595797533806) q[4];
ry(-0.18324470687662506) q[5];
rz(1.4539774450276965) q[5];
ry(-3.1383114633359654) q[6];
rz(0.02259599403001644) q[6];
ry(3.117472970338209) q[7];
rz(-1.2633236583872582) q[7];
ry(-0.7483244970913644) q[8];
rz(2.4319832790759874) q[8];
ry(0.22423154133480375) q[9];
rz(-1.221843126019547) q[9];
ry(-2.956743232035349) q[10];
rz(-0.6868931330137268) q[10];
ry(1.5897969678561834) q[11];
rz(0.3502489290349449) q[11];
ry(-2.9336283176380133) q[12];
rz(-3.093581222980467) q[12];
ry(2.234920318226938) q[13];
rz(-2.2067448414777777) q[13];
ry(-2.8773892023552676) q[14];
rz(2.048013703969742) q[14];
ry(-0.20629201015704907) q[15];
rz(1.3635908055644368) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.5865098474500687) q[0];
rz(-1.7892908632740863) q[0];
ry(0.0486398518646757) q[1];
rz(-2.5096217965625076) q[1];
ry(2.7055797740707757) q[2];
rz(-2.208968888841235) q[2];
ry(0.17189559416904138) q[3];
rz(-0.42820975288162516) q[3];
ry(0.9507575825918897) q[4];
rz(-2.742012396342915) q[4];
ry(2.5340741113714507) q[5];
rz(-1.3392653334720839) q[5];
ry(-0.9473780360673707) q[6];
rz(1.389832612785411) q[6];
ry(-2.3804744897760917) q[7];
rz(-1.3317467490555988) q[7];
ry(-0.007070221993408589) q[8];
rz(2.340878159522047) q[8];
ry(0.5721238565508333) q[9];
rz(2.626362502188627) q[9];
ry(-1.341410512418067) q[10];
rz(-3.122580027855646) q[10];
ry(-2.5635508243343006) q[11];
rz(1.3891597784245668) q[11];
ry(-1.6124943303289778) q[12];
rz(1.850688820283208) q[12];
ry(-1.6088716414038222) q[13];
rz(-2.6967897183048826) q[13];
ry(-0.8414489502986531) q[14];
rz(-0.8915417447972764) q[14];
ry(-2.6244917538189227) q[15];
rz(2.2898130620409685) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.761010313943513) q[0];
rz(0.5709883676452796) q[0];
ry(-2.5312111024737023) q[1];
rz(-2.6446418659167454) q[1];
ry(-1.1935512992514334) q[2];
rz(-0.5823239255841592) q[2];
ry(2.589645440703313) q[3];
rz(2.9841919869325526) q[3];
ry(-0.053415851218362675) q[4];
rz(0.9735502452788579) q[4];
ry(3.124256334170928) q[5];
rz(-1.396420105194407) q[5];
ry(0.008848132112301954) q[6];
rz(0.5675728414070821) q[6];
ry(3.002042171774403) q[7];
rz(-3.0412307889429484) q[7];
ry(-0.10802158237338855) q[8];
rz(-2.077953909676703) q[8];
ry(3.014904150317003) q[9];
rz(0.06624093484372295) q[9];
ry(0.05284164030766547) q[10];
rz(2.3888540508983462) q[10];
ry(-0.009816558873946057) q[11];
rz(3.0294136726621903) q[11];
ry(3.1375534590258263) q[12];
rz(-2.570574953359111) q[12];
ry(1.5960774893923686) q[13];
rz(1.9402920342935221) q[13];
ry(0.774345850367675) q[14];
rz(1.0414708717151129) q[14];
ry(-1.5306905031322573) q[15];
rz(-1.2034008469735307) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.5207183331402625) q[0];
rz(-2.1473767911677784) q[0];
ry(2.539598683695384) q[1];
rz(2.813219357999977) q[1];
ry(-0.2890518858809914) q[2];
rz(-1.3135396627842342) q[2];
ry(-1.4466657974110486) q[3];
rz(0.12552279689786605) q[3];
ry(-0.04604009429259648) q[4];
rz(-2.71315617541919) q[4];
ry(-1.1061729071623716) q[5];
rz(2.5517389259900383) q[5];
ry(2.118125735978839) q[6];
rz(-2.73133479929472) q[6];
ry(0.6359780635515486) q[7];
rz(-2.4220013374525453) q[7];
ry(-3.1230018193124214) q[8];
rz(-0.6174536107800384) q[8];
ry(0.5431095030128527) q[9];
rz(-0.9021347489426618) q[9];
ry(-1.6088164276519636) q[10];
rz(1.9369178765970179) q[10];
ry(2.606019727552113) q[11];
rz(1.625720773476508) q[11];
ry(-1.3154689124662093) q[12];
rz(-0.12376937624100143) q[12];
ry(2.7768244860511735) q[13];
rz(2.412011503969366) q[13];
ry(1.9314979401738501) q[14];
rz(-0.08468750642769163) q[14];
ry(2.739668048446611) q[15];
rz(-2.6303212269576415) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.5476412321139977) q[0];
rz(-0.27070455229504553) q[0];
ry(2.749169188812697) q[1];
rz(-1.4657932164802547) q[1];
ry(3.1187540523381334) q[2];
rz(2.3142421885019284) q[2];
ry(2.6965081310239425) q[3];
rz(2.4299118352638764) q[3];
ry(-3.1114967134968388) q[4];
rz(1.1994410713570245) q[4];
ry(-3.1016530769378456) q[5];
rz(1.7883849946944723) q[5];
ry(-0.025026012816157975) q[6];
rz(2.7175911067109326) q[6];
ry(-3.0705000751785323) q[7];
rz(-1.0039669619743607) q[7];
ry(1.600407818453685) q[8];
rz(1.4831690574342298) q[8];
ry(-0.10438113261400317) q[9];
rz(1.0882828901756376) q[9];
ry(-2.6209104609733727) q[10];
rz(-2.8937041291209646) q[10];
ry(-3.0962416940176) q[11];
rz(-0.7838578725596953) q[11];
ry(-0.014500146079836185) q[12];
rz(1.9269685008889899) q[12];
ry(-0.08692762320841128) q[13];
rz(-2.351562936823977) q[13];
ry(1.4154523998775161) q[14];
rz(-2.5125692321558963) q[14];
ry(-2.4642644467511845) q[15];
rz(-0.45875021720636) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.092781237785929) q[0];
rz(-1.26565464673147) q[0];
ry(-1.3257298536743738) q[1];
rz(2.1585991867161276) q[1];
ry(2.416352874456396) q[2];
rz(2.317044292594774) q[2];
ry(-1.6565453514033486) q[3];
rz(-1.4834472436267936) q[3];
ry(1.7592089555643264) q[4];
rz(0.5500293050283308) q[4];
ry(-1.865295199167934) q[5];
rz(-1.3796960256518398) q[5];
ry(2.0766366077243634) q[6];
rz(-2.6492833303499363) q[6];
ry(1.3741457714432364) q[7];
rz(-2.2328525231794485) q[7];
ry(-2.183962270444873) q[8];
rz(-0.29342668141568495) q[8];
ry(0.0017809002626769657) q[9];
rz(2.7326786046092644) q[9];
ry(-2.6238343006547176) q[10];
rz(-2.7898365842644828) q[10];
ry(-0.9633826673491299) q[11];
rz(1.0942228545242685) q[11];
ry(-0.3103482775664204) q[12];
rz(0.3081987212258053) q[12];
ry(-1.8293692833731798) q[13];
rz(-1.0237359962216683) q[13];
ry(1.2334816262188857) q[14];
rz(1.1122627661303564) q[14];
ry(1.8136419091834792) q[15];
rz(-1.5739594690331318) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.5361631143365839) q[0];
rz(-0.49071468535540597) q[0];
ry(0.2082782214867951) q[1];
rz(-2.861101209603388) q[1];
ry(-2.997699600283608) q[2];
rz(0.10879371512092804) q[2];
ry(-0.10997699219073365) q[3];
rz(2.6081846588430486) q[3];
ry(3.080914517369022) q[4];
rz(-0.4508497423181969) q[4];
ry(-3.1294293964715503) q[5];
rz(1.8071210635953898) q[5];
ry(3.0973261024973535) q[6];
rz(-2.9824600718176173) q[6];
ry(-0.0030016778332537797) q[7];
rz(3.029321778470889) q[7];
ry(0.3564209652393835) q[8];
rz(-3.03237153242719) q[8];
ry(3.020126061114216) q[9];
rz(-0.061310162131292145) q[9];
ry(-0.28533943225969305) q[10];
rz(-1.7549644132711109) q[10];
ry(0.07219384085561664) q[11];
rz(2.028957660615011) q[11];
ry(3.134390348312801) q[12];
rz(-0.87763659420379) q[12];
ry(-3.1023773214498567) q[13];
rz(2.9494634621727283) q[13];
ry(2.6510589029554272) q[14];
rz(0.6193470637050682) q[14];
ry(1.3244140787381404) q[15];
rz(0.153246149962742) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.107635558036945) q[0];
rz(1.3123851192185942) q[0];
ry(-1.3410053566760045) q[1];
rz(-1.540677871783241) q[1];
ry(-1.473817180976087) q[2];
rz(-1.6659808602499524) q[2];
ry(0.7602634322605507) q[3];
rz(-0.9032388142215425) q[3];
ry(0.7870051998455718) q[4];
rz(1.6126468935181233) q[4];
ry(-1.7268878917459833) q[5];
rz(-1.7886324191906717) q[5];
ry(2.343531777285232) q[6];
rz(-1.3424173185141741) q[6];
ry(-0.10789259460914392) q[7];
rz(-2.017768519708067) q[7];
ry(0.6121603570583869) q[8];
rz(-2.2413011289942637) q[8];
ry(-1.566348774919561) q[9];
rz(0.7715123666527414) q[9];
ry(-1.521608197499777) q[10];
rz(-1.9021363593222833) q[10];
ry(3.101340074525164) q[11];
rz(2.892205849604237) q[11];
ry(-2.927491903869323) q[12];
rz(-3.071834613669162) q[12];
ry(2.6821608482063857) q[13];
rz(0.7145052659521792) q[13];
ry(-2.970512411772737) q[14];
rz(-1.9437338053075415) q[14];
ry(0.49868881833221934) q[15];
rz(0.26576907885307927) q[15];