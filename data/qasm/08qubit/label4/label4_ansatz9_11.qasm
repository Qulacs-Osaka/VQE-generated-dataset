OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.797098087683125) q[0];
ry(-1.2911198965121375) q[1];
cx q[0],q[1];
ry(-0.5254121722163764) q[0];
ry(-0.23541560810110845) q[1];
cx q[0],q[1];
ry(-1.0159802737285926) q[2];
ry(-2.041649587052199) q[3];
cx q[2],q[3];
ry(-2.832559497320785) q[2];
ry(1.6672248387911526) q[3];
cx q[2],q[3];
ry(0.9558519058531321) q[4];
ry(1.5644398961361543) q[5];
cx q[4],q[5];
ry(0.5985095434103356) q[4];
ry(2.0671985592962363) q[5];
cx q[4],q[5];
ry(2.0507923744034473) q[6];
ry(0.7394621818439048) q[7];
cx q[6],q[7];
ry(-1.9534287146231122) q[6];
ry(3.0202340541468806) q[7];
cx q[6],q[7];
ry(2.488116908167225) q[0];
ry(-2.199537204355429) q[2];
cx q[0],q[2];
ry(1.6685918717319375) q[0];
ry(-2.072244711250717) q[2];
cx q[0],q[2];
ry(-0.32957945664909966) q[2];
ry(2.4981406248241065) q[4];
cx q[2],q[4];
ry(-0.11791354637500095) q[2];
ry(-0.17312619311679553) q[4];
cx q[2],q[4];
ry(-0.7523877534586481) q[4];
ry(0.49686329124929607) q[6];
cx q[4],q[6];
ry(1.8070181923483504) q[4];
ry(-1.4460522523058534) q[6];
cx q[4],q[6];
ry(-0.4694779357119584) q[1];
ry(2.122934022152067) q[3];
cx q[1],q[3];
ry(-1.141741566702101) q[1];
ry(1.232333896926952) q[3];
cx q[1],q[3];
ry(1.4194166013758105) q[3];
ry(2.8424997747474774) q[5];
cx q[3],q[5];
ry(1.4835310573798448) q[3];
ry(0.4585644265310789) q[5];
cx q[3],q[5];
ry(-2.94170856941413) q[5];
ry(2.1941018223663367) q[7];
cx q[5],q[7];
ry(0.317938710250813) q[5];
ry(0.519446278789708) q[7];
cx q[5],q[7];
ry(1.4567353819287414) q[0];
ry(0.802046998962779) q[3];
cx q[0],q[3];
ry(-0.783577132721363) q[0];
ry(-1.8924643402150743) q[3];
cx q[0],q[3];
ry(-0.45281315107260234) q[1];
ry(-1.1087326304809162) q[2];
cx q[1],q[2];
ry(0.3186030213835487) q[1];
ry(1.4304164032413895) q[2];
cx q[1],q[2];
ry(-2.4980459226605602) q[2];
ry(1.9684216877630645) q[5];
cx q[2],q[5];
ry(-2.6778814981633228) q[2];
ry(1.7113584167172797) q[5];
cx q[2],q[5];
ry(1.9665827245108998) q[3];
ry(1.6709045570243621) q[4];
cx q[3],q[4];
ry(-1.1497400506132167) q[3];
ry(2.958138227728126) q[4];
cx q[3],q[4];
ry(-1.224047137049288) q[4];
ry(1.8857667698903748) q[7];
cx q[4],q[7];
ry(-2.596193737407814) q[4];
ry(-1.9568141865257633) q[7];
cx q[4],q[7];
ry(-1.6010953379817146) q[5];
ry(0.843447310444176) q[6];
cx q[5],q[6];
ry(-1.3119526874437344) q[5];
ry(-2.24110717508736) q[6];
cx q[5],q[6];
ry(1.4381743419188249) q[0];
ry(-2.441893673528526) q[1];
cx q[0],q[1];
ry(1.0112143435810015) q[0];
ry(-0.18796530581164078) q[1];
cx q[0],q[1];
ry(-2.0683959166350365) q[2];
ry(2.2874903956710266) q[3];
cx q[2],q[3];
ry(-0.8641666550712807) q[2];
ry(3.130317985165962) q[3];
cx q[2],q[3];
ry(2.9866981179602217) q[4];
ry(-1.4251488715473444) q[5];
cx q[4],q[5];
ry(-2.7679547817295944) q[4];
ry(-0.9486660844783854) q[5];
cx q[4],q[5];
ry(-1.5176936696916878) q[6];
ry(0.1198851624578028) q[7];
cx q[6],q[7];
ry(2.7695553178110424) q[6];
ry(1.9915941623026754) q[7];
cx q[6],q[7];
ry(1.1966446635134975) q[0];
ry(-0.21882484382001363) q[2];
cx q[0],q[2];
ry(-1.7389367193437117) q[0];
ry(-2.8634868720339797) q[2];
cx q[0],q[2];
ry(-1.242543622934837) q[2];
ry(-0.33034701911974734) q[4];
cx q[2],q[4];
ry(-0.02690985502642906) q[2];
ry(0.3512995418542573) q[4];
cx q[2],q[4];
ry(-1.690613000700008) q[4];
ry(2.881421372435663) q[6];
cx q[4],q[6];
ry(-1.486011756744654) q[4];
ry(2.6597763327526645) q[6];
cx q[4],q[6];
ry(-0.4391618572465151) q[1];
ry(2.2813298376522195) q[3];
cx q[1],q[3];
ry(-1.699170472652838) q[1];
ry(2.509640326154831) q[3];
cx q[1],q[3];
ry(-1.1178729337191298) q[3];
ry(-1.0303549473474058) q[5];
cx q[3],q[5];
ry(1.5131044620157468) q[3];
ry(2.794101334761276) q[5];
cx q[3],q[5];
ry(1.7057052061338398) q[5];
ry(1.4224035158246178) q[7];
cx q[5],q[7];
ry(2.2657965869535284) q[5];
ry(2.269957769709438) q[7];
cx q[5],q[7];
ry(2.876644770643373) q[0];
ry(1.3381134788460476) q[3];
cx q[0],q[3];
ry(2.086281777686551) q[0];
ry(-1.5554395144923328) q[3];
cx q[0],q[3];
ry(-2.787776988523418) q[1];
ry(-2.6946943499145823) q[2];
cx q[1],q[2];
ry(-2.9823374180608546) q[1];
ry(-2.6814373513993113) q[2];
cx q[1],q[2];
ry(-2.3114821955422724) q[2];
ry(1.1887492203960621) q[5];
cx q[2],q[5];
ry(2.7863085460521977) q[2];
ry(-0.2565688274513471) q[5];
cx q[2],q[5];
ry(-2.5386517316055732) q[3];
ry(-2.8306893379538773) q[4];
cx q[3],q[4];
ry(-1.1820855339426686) q[3];
ry(0.5539123874582907) q[4];
cx q[3],q[4];
ry(2.2027124561670792) q[4];
ry(1.320700789017148) q[7];
cx q[4],q[7];
ry(-3.070291216279391) q[4];
ry(1.3032052327400168) q[7];
cx q[4],q[7];
ry(-1.9784085971783902) q[5];
ry(-2.812509065442003) q[6];
cx q[5],q[6];
ry(-1.9582841361060446) q[5];
ry(-2.4399686684210944) q[6];
cx q[5],q[6];
ry(0.7578719669149692) q[0];
ry(2.955121325570259) q[1];
cx q[0],q[1];
ry(-2.0530755096314808) q[0];
ry(0.8697873638324171) q[1];
cx q[0],q[1];
ry(-1.589520436414714) q[2];
ry(-0.37718679469920446) q[3];
cx q[2],q[3];
ry(1.3790176900949453) q[2];
ry(-2.299874075170534) q[3];
cx q[2],q[3];
ry(2.8997177053941883) q[4];
ry(2.3100119643450943) q[5];
cx q[4],q[5];
ry(-0.44168661999711034) q[4];
ry(-1.941159632706837) q[5];
cx q[4],q[5];
ry(-2.8334789123099915) q[6];
ry(-0.739477088136848) q[7];
cx q[6],q[7];
ry(1.8490196246485235) q[6];
ry(-2.155941938870466) q[7];
cx q[6],q[7];
ry(0.46216830811736376) q[0];
ry(-1.175896881043692) q[2];
cx q[0],q[2];
ry(-1.7924789988971657) q[0];
ry(-0.6369269727840324) q[2];
cx q[0],q[2];
ry(1.6888714689410929) q[2];
ry(0.9138585400177432) q[4];
cx q[2],q[4];
ry(-1.7657820361713172) q[2];
ry(-2.6292886398100364) q[4];
cx q[2],q[4];
ry(2.5755630717988693) q[4];
ry(-0.682194784065717) q[6];
cx q[4],q[6];
ry(-2.8329714621092514) q[4];
ry(-2.4193799847513775) q[6];
cx q[4],q[6];
ry(2.3638422569494004) q[1];
ry(-2.481885958931209) q[3];
cx q[1],q[3];
ry(-1.3368288800922885) q[1];
ry(-1.9491360089186696) q[3];
cx q[1],q[3];
ry(-0.05427335422454924) q[3];
ry(0.3309698546295775) q[5];
cx q[3],q[5];
ry(2.3682461291591212) q[3];
ry(1.618788610693077) q[5];
cx q[3],q[5];
ry(-1.0196631706939812) q[5];
ry(2.975178408307222) q[7];
cx q[5],q[7];
ry(-2.088353740502928) q[5];
ry(0.4770254493979653) q[7];
cx q[5],q[7];
ry(-2.80566320858619) q[0];
ry(1.9597903767358713) q[3];
cx q[0],q[3];
ry(-0.12872867347171285) q[0];
ry(-0.6622356875116333) q[3];
cx q[0],q[3];
ry(1.5313230939692974) q[1];
ry(1.0427767413962217) q[2];
cx q[1],q[2];
ry(2.084055460204032) q[1];
ry(0.40016245821530394) q[2];
cx q[1],q[2];
ry(0.20908189430977142) q[2];
ry(-2.573584561717717) q[5];
cx q[2],q[5];
ry(-0.6629385379570334) q[2];
ry(0.3740284681034379) q[5];
cx q[2],q[5];
ry(2.357627031373744) q[3];
ry(-0.8007556793567074) q[4];
cx q[3],q[4];
ry(3.0465446223777826) q[3];
ry(-2.683241890171834) q[4];
cx q[3],q[4];
ry(0.22619904782832467) q[4];
ry(3.0144065584258777) q[7];
cx q[4],q[7];
ry(-1.7368736856303804) q[4];
ry(-1.0221457636411335) q[7];
cx q[4],q[7];
ry(1.1196514258086399) q[5];
ry(1.4689037982853395) q[6];
cx q[5],q[6];
ry(0.6539861761031286) q[5];
ry(-1.9553444720173807) q[6];
cx q[5],q[6];
ry(-0.7558377842667624) q[0];
ry(2.8174977864882726) q[1];
cx q[0],q[1];
ry(-1.3808581351348808) q[0];
ry(1.4717132210057704) q[1];
cx q[0],q[1];
ry(-1.6907369947320365) q[2];
ry(-2.4767156137155197) q[3];
cx q[2],q[3];
ry(-1.9249501273510585) q[2];
ry(0.36648356263852894) q[3];
cx q[2],q[3];
ry(-1.5229979282149453) q[4];
ry(2.4543560852351827) q[5];
cx q[4],q[5];
ry(1.7707471696284047) q[4];
ry(-0.8176513835716914) q[5];
cx q[4],q[5];
ry(-0.6683569588168171) q[6];
ry(-0.9669577208134904) q[7];
cx q[6],q[7];
ry(2.9878700631166946) q[6];
ry(-1.000695468756188) q[7];
cx q[6],q[7];
ry(-2.955612408764563) q[0];
ry(-3.003533132325497) q[2];
cx q[0],q[2];
ry(-1.3635599079203269) q[0];
ry(2.1601509352068367) q[2];
cx q[0],q[2];
ry(-0.1975338083843485) q[2];
ry(2.8133967503772723) q[4];
cx q[2],q[4];
ry(1.4822635326939348) q[2];
ry(2.1465854832160285) q[4];
cx q[2],q[4];
ry(0.5118133595656369) q[4];
ry(-0.4274747739027198) q[6];
cx q[4],q[6];
ry(-3.125558878980286) q[4];
ry(0.7153100459055743) q[6];
cx q[4],q[6];
ry(1.268182964355824) q[1];
ry(-2.652277721872172) q[3];
cx q[1],q[3];
ry(-0.6039375290748974) q[1];
ry(-1.8817870038630022) q[3];
cx q[1],q[3];
ry(-0.9156851066501615) q[3];
ry(-1.8268659259806106) q[5];
cx q[3],q[5];
ry(2.969458512087302) q[3];
ry(-0.7776838146392046) q[5];
cx q[3],q[5];
ry(-0.5820709341401923) q[5];
ry(-1.6399662612402084) q[7];
cx q[5],q[7];
ry(2.936867130661123) q[5];
ry(0.917479326254071) q[7];
cx q[5],q[7];
ry(1.8992977197187386) q[0];
ry(0.40755646555362895) q[3];
cx q[0],q[3];
ry(2.592620678876677) q[0];
ry(-2.7233336163423822) q[3];
cx q[0],q[3];
ry(-1.4064293401801207) q[1];
ry(-2.1801376726766835) q[2];
cx q[1],q[2];
ry(-1.9177428135327919) q[1];
ry(-2.8656820710171855) q[2];
cx q[1],q[2];
ry(-0.9884230002548758) q[2];
ry(2.311879561255381) q[5];
cx q[2],q[5];
ry(-2.3241185326142033) q[2];
ry(-2.3369956827724963) q[5];
cx q[2],q[5];
ry(-3.046917257651749) q[3];
ry(-0.8757478069965212) q[4];
cx q[3],q[4];
ry(-0.1433362407079164) q[3];
ry(-1.2319643841708539) q[4];
cx q[3],q[4];
ry(-2.350125820377711) q[4];
ry(-1.0974026694702594) q[7];
cx q[4],q[7];
ry(-1.006999245773426) q[4];
ry(2.0036585416681794) q[7];
cx q[4],q[7];
ry(0.520031777905837) q[5];
ry(-1.1850626680756635) q[6];
cx q[5],q[6];
ry(-1.362466468285401) q[5];
ry(-3.1353528622036153) q[6];
cx q[5],q[6];
ry(-1.7355907511817055) q[0];
ry(-3.1271535392930536) q[1];
cx q[0],q[1];
ry(1.809055667976243) q[0];
ry(-1.4589986238100252) q[1];
cx q[0],q[1];
ry(-1.7673326200033577) q[2];
ry(2.413419572614599) q[3];
cx q[2],q[3];
ry(2.2008182332112005) q[2];
ry(1.1523433768538842) q[3];
cx q[2],q[3];
ry(2.8693814430159907) q[4];
ry(-2.3555759537858054) q[5];
cx q[4],q[5];
ry(-1.5670019042646952) q[4];
ry(-1.5064005165416472) q[5];
cx q[4],q[5];
ry(-0.10505035963210196) q[6];
ry(2.3231232475574837) q[7];
cx q[6],q[7];
ry(-1.352968494015715) q[6];
ry(-0.519098587704768) q[7];
cx q[6],q[7];
ry(-0.20151138537703925) q[0];
ry(-2.950531516529845) q[2];
cx q[0],q[2];
ry(-1.0356986136360098) q[0];
ry(0.7340719327465672) q[2];
cx q[0],q[2];
ry(-1.1137606725740514) q[2];
ry(1.6115220407178559) q[4];
cx q[2],q[4];
ry(-2.2523539489006184) q[2];
ry(1.4700501939855632) q[4];
cx q[2],q[4];
ry(0.6352280832241872) q[4];
ry(-1.8795326379157862) q[6];
cx q[4],q[6];
ry(0.6482712629693638) q[4];
ry(1.4909736158037612) q[6];
cx q[4],q[6];
ry(-2.021573947673504) q[1];
ry(-1.1711661851474602) q[3];
cx q[1],q[3];
ry(-0.6683246855364233) q[1];
ry(1.3399952903781143) q[3];
cx q[1],q[3];
ry(2.723789388784356) q[3];
ry(0.5642848039865997) q[5];
cx q[3],q[5];
ry(-0.11006921403584437) q[3];
ry(1.199514090110707) q[5];
cx q[3],q[5];
ry(-2.3881493564444165) q[5];
ry(-0.7574678728540979) q[7];
cx q[5],q[7];
ry(-2.8156430566318256) q[5];
ry(-0.7265005068847986) q[7];
cx q[5],q[7];
ry(-2.9307662379378927) q[0];
ry(0.6389249076191786) q[3];
cx q[0],q[3];
ry(2.9146212713417863) q[0];
ry(0.492529470483495) q[3];
cx q[0],q[3];
ry(-0.6865831273890488) q[1];
ry(2.848628556449713) q[2];
cx q[1],q[2];
ry(2.740931531193232) q[1];
ry(2.2158827277822337) q[2];
cx q[1],q[2];
ry(-3.12113788278632) q[2];
ry(3.100127356539716) q[5];
cx q[2],q[5];
ry(-2.7283856979211687) q[2];
ry(1.930335474456351) q[5];
cx q[2],q[5];
ry(-1.6593035011959651) q[3];
ry(2.3050168935083137) q[4];
cx q[3],q[4];
ry(0.5146203817101389) q[3];
ry(-2.5257833982003164) q[4];
cx q[3],q[4];
ry(-0.6174671710656553) q[4];
ry(0.3748301080911922) q[7];
cx q[4],q[7];
ry(3.1023086451821045) q[4];
ry(0.5848870818296823) q[7];
cx q[4],q[7];
ry(-1.6921241488683456) q[5];
ry(-1.526853830973085) q[6];
cx q[5],q[6];
ry(2.660589471727482) q[5];
ry(-2.646786610894001) q[6];
cx q[5],q[6];
ry(-0.1466506314142093) q[0];
ry(1.1679611415241773) q[1];
cx q[0],q[1];
ry(-1.7082990000827873) q[0];
ry(-1.579538107885454) q[1];
cx q[0],q[1];
ry(-2.6685935455326053) q[2];
ry(-1.3614935396259131) q[3];
cx q[2],q[3];
ry(-0.36010053758967936) q[2];
ry(1.242828423133966) q[3];
cx q[2],q[3];
ry(-1.7465004247344629) q[4];
ry(-0.3523634262171403) q[5];
cx q[4],q[5];
ry(1.22066104886932) q[4];
ry(1.7646300624898255) q[5];
cx q[4],q[5];
ry(-2.005734272058751) q[6];
ry(0.7583961643270403) q[7];
cx q[6],q[7];
ry(-0.8306091413143863) q[6];
ry(0.9458812049031193) q[7];
cx q[6],q[7];
ry(2.684684105916999) q[0];
ry(-1.5426699948981577) q[2];
cx q[0],q[2];
ry(0.6675789851610966) q[0];
ry(1.493530809550238) q[2];
cx q[0],q[2];
ry(-3.0853963345010267) q[2];
ry(0.9294132428250822) q[4];
cx q[2],q[4];
ry(0.5065778120564632) q[2];
ry(0.1717484153555402) q[4];
cx q[2],q[4];
ry(1.1858567946067353) q[4];
ry(1.7236785436429838) q[6];
cx q[4],q[6];
ry(1.8054943273441104) q[4];
ry(-0.7662409445857926) q[6];
cx q[4],q[6];
ry(0.9053897555443706) q[1];
ry(-0.9566947826840322) q[3];
cx q[1],q[3];
ry(0.8718790339837046) q[1];
ry(-2.1888928619281565) q[3];
cx q[1],q[3];
ry(-2.92777179075922) q[3];
ry(-0.586727477461511) q[5];
cx q[3],q[5];
ry(1.7118505152967851) q[3];
ry(-2.83059683079448) q[5];
cx q[3],q[5];
ry(0.0336008324342556) q[5];
ry(-1.2206867757890472) q[7];
cx q[5],q[7];
ry(0.1117714023391949) q[5];
ry(-1.0355836023770355) q[7];
cx q[5],q[7];
ry(2.5872891124546085) q[0];
ry(-1.4038810926180494) q[3];
cx q[0],q[3];
ry(2.623045816041235) q[0];
ry(-2.170772513637765) q[3];
cx q[0],q[3];
ry(-2.615218393521762) q[1];
ry(2.806395725114941) q[2];
cx q[1],q[2];
ry(-2.1323146571942324) q[1];
ry(-2.063323687028734) q[2];
cx q[1],q[2];
ry(1.694832319929488) q[2];
ry(0.5987005096402641) q[5];
cx q[2],q[5];
ry(-0.776431395052561) q[2];
ry(-2.6155022677621935) q[5];
cx q[2],q[5];
ry(0.22089003227773463) q[3];
ry(-1.007159203329146) q[4];
cx q[3],q[4];
ry(-2.7559528544152587) q[3];
ry(3.0648174437774847) q[4];
cx q[3],q[4];
ry(-1.0000549668689702) q[4];
ry(-2.220057591710032) q[7];
cx q[4],q[7];
ry(-0.13288777649383565) q[4];
ry(-2.1438291355479597) q[7];
cx q[4],q[7];
ry(-1.050995549755613) q[5];
ry(1.3811320049852038) q[6];
cx q[5],q[6];
ry(2.359774893356583) q[5];
ry(-0.289428907133993) q[6];
cx q[5],q[6];
ry(-1.7535676969996585) q[0];
ry(-2.6472095096194543) q[1];
cx q[0],q[1];
ry(2.529285803041883) q[0];
ry(1.767969433506076) q[1];
cx q[0],q[1];
ry(-2.026750658925981) q[2];
ry(-0.03997636491742337) q[3];
cx q[2],q[3];
ry(-1.9501750555018666) q[2];
ry(-0.0009477673316817813) q[3];
cx q[2],q[3];
ry(-0.5024266975935793) q[4];
ry(0.9660020791131787) q[5];
cx q[4],q[5];
ry(-1.4638519384104685) q[4];
ry(-0.733528689890329) q[5];
cx q[4],q[5];
ry(0.17304000399227554) q[6];
ry(-1.4983753635868649) q[7];
cx q[6],q[7];
ry(-1.3822300784688106) q[6];
ry(-0.7381975125187088) q[7];
cx q[6],q[7];
ry(0.43584425295714413) q[0];
ry(-2.160617351193113) q[2];
cx q[0],q[2];
ry(-2.033606661191226) q[0];
ry(-0.5544379383932831) q[2];
cx q[0],q[2];
ry(-0.8990594950403538) q[2];
ry(0.6750570774930189) q[4];
cx q[2],q[4];
ry(-0.5252074288181898) q[2];
ry(-0.05142276673755841) q[4];
cx q[2],q[4];
ry(2.049988811634141) q[4];
ry(2.005802911055723) q[6];
cx q[4],q[6];
ry(-0.8033660160053985) q[4];
ry(-2.7124636747813047) q[6];
cx q[4],q[6];
ry(0.019556483999046744) q[1];
ry(2.877968018783253) q[3];
cx q[1],q[3];
ry(-1.7958841130500751) q[1];
ry(2.2410837522747857) q[3];
cx q[1],q[3];
ry(-0.5258709967688437) q[3];
ry(0.8150489586319063) q[5];
cx q[3],q[5];
ry(1.8471417890961916) q[3];
ry(-1.8870070031616164) q[5];
cx q[3],q[5];
ry(-1.4818344042348572) q[5];
ry(0.5154112644657783) q[7];
cx q[5],q[7];
ry(-3.0452954722615506) q[5];
ry(1.2422641406746056) q[7];
cx q[5],q[7];
ry(1.436743078624884) q[0];
ry(1.1033612716844048) q[3];
cx q[0],q[3];
ry(0.2643696585256804) q[0];
ry(3.0670871621967253) q[3];
cx q[0],q[3];
ry(0.05983459828640302) q[1];
ry(1.4160248976672012) q[2];
cx q[1],q[2];
ry(2.6702961972148618) q[1];
ry(-2.1155310241648833) q[2];
cx q[1],q[2];
ry(-0.6380285371159656) q[2];
ry(2.4696931383223513) q[5];
cx q[2],q[5];
ry(-1.9390067643978646) q[2];
ry(-2.102221678677238) q[5];
cx q[2],q[5];
ry(1.7454402343821158) q[3];
ry(-1.495792933066852) q[4];
cx q[3],q[4];
ry(2.4817949662779446) q[3];
ry(-2.24411175687902) q[4];
cx q[3],q[4];
ry(2.807058253113879) q[4];
ry(1.7200467214789823) q[7];
cx q[4],q[7];
ry(-1.841544338024977) q[4];
ry(2.229098289558989) q[7];
cx q[4],q[7];
ry(-1.3982990886397886) q[5];
ry(-0.9419717621481271) q[6];
cx q[5],q[6];
ry(-0.2049564104204622) q[5];
ry(-1.7388166029758063) q[6];
cx q[5],q[6];
ry(-2.1074738187705955) q[0];
ry(-0.8261663382309463) q[1];
cx q[0],q[1];
ry(1.997034386486943) q[0];
ry(0.8530705129960408) q[1];
cx q[0],q[1];
ry(-0.7599454580713605) q[2];
ry(-2.8796326486714086) q[3];
cx q[2],q[3];
ry(1.7397124089875238) q[2];
ry(1.2748221882518975) q[3];
cx q[2],q[3];
ry(1.0235240878694094) q[4];
ry(-1.3961794715765166) q[5];
cx q[4],q[5];
ry(0.04268417733123239) q[4];
ry(-2.06102256754349) q[5];
cx q[4],q[5];
ry(-0.09764073199832879) q[6];
ry(-2.5669643570008076) q[7];
cx q[6],q[7];
ry(3.1000716295087023) q[6];
ry(2.0942718835548204) q[7];
cx q[6],q[7];
ry(-0.19597332855246694) q[0];
ry(-2.648791486092034) q[2];
cx q[0],q[2];
ry(2.1246076894133576) q[0];
ry(-2.0370251429726043) q[2];
cx q[0],q[2];
ry(-1.861651987556259) q[2];
ry(0.8562099288228431) q[4];
cx q[2],q[4];
ry(-1.0411554152861213) q[2];
ry(-1.8338003617352563) q[4];
cx q[2],q[4];
ry(2.2205642882216727) q[4];
ry(0.2250941157171873) q[6];
cx q[4],q[6];
ry(2.7013696337918285) q[4];
ry(1.8327914133466703) q[6];
cx q[4],q[6];
ry(2.5385326862223403) q[1];
ry(1.343047299897372) q[3];
cx q[1],q[3];
ry(3.1200124680754193) q[1];
ry(0.30308127060489465) q[3];
cx q[1],q[3];
ry(-2.7896648633673258) q[3];
ry(-2.6931515456334534) q[5];
cx q[3],q[5];
ry(2.3555324920193867) q[3];
ry(-1.5853917890210267) q[5];
cx q[3],q[5];
ry(-1.802229900805552) q[5];
ry(0.4820179443671133) q[7];
cx q[5],q[7];
ry(1.099167539689046) q[5];
ry(-0.23763740047578227) q[7];
cx q[5],q[7];
ry(1.3509606556040978) q[0];
ry(0.7420887961380247) q[3];
cx q[0],q[3];
ry(0.6125987209757008) q[0];
ry(-3.1106884337398433) q[3];
cx q[0],q[3];
ry(1.846974161313632) q[1];
ry(-2.926439572694737) q[2];
cx q[1],q[2];
ry(-1.2658043148738862) q[1];
ry(0.27595173557172353) q[2];
cx q[1],q[2];
ry(2.7926532348066004) q[2];
ry(-1.540970412992984) q[5];
cx q[2],q[5];
ry(0.09668002468872301) q[2];
ry(2.6235591276205965) q[5];
cx q[2],q[5];
ry(0.058307340609908394) q[3];
ry(0.980363625990562) q[4];
cx q[3],q[4];
ry(-0.9621371526829386) q[3];
ry(2.780413571635301) q[4];
cx q[3],q[4];
ry(0.6878492748937525) q[4];
ry(0.2048226995244384) q[7];
cx q[4],q[7];
ry(1.40523796032406) q[4];
ry(2.0209030929606167) q[7];
cx q[4],q[7];
ry(-2.9373038462841623) q[5];
ry(0.8592130320361406) q[6];
cx q[5],q[6];
ry(1.7913318755933416) q[5];
ry(0.22042909357833965) q[6];
cx q[5],q[6];
ry(2.8709695401158024) q[0];
ry(2.285873066227373) q[1];
cx q[0],q[1];
ry(-1.586859443721863) q[0];
ry(-1.7205321731858634) q[1];
cx q[0],q[1];
ry(0.835165826040801) q[2];
ry(0.11846346694302579) q[3];
cx q[2],q[3];
ry(-0.6990194601322344) q[2];
ry(0.7566388542525928) q[3];
cx q[2],q[3];
ry(-0.7253391272829655) q[4];
ry(1.5184550546616178) q[5];
cx q[4],q[5];
ry(3.120780844304763) q[4];
ry(1.968698350959265) q[5];
cx q[4],q[5];
ry(-2.1582079940536136) q[6];
ry(1.364373961986634) q[7];
cx q[6],q[7];
ry(1.0603526576338984) q[6];
ry(2.499906423900803) q[7];
cx q[6],q[7];
ry(-3.065666771618602) q[0];
ry(-2.84561315868186) q[2];
cx q[0],q[2];
ry(2.7933526953655723) q[0];
ry(-0.3824028694423207) q[2];
cx q[0],q[2];
ry(1.1592750535911431) q[2];
ry(-1.3616611204378382) q[4];
cx q[2],q[4];
ry(-2.428120962253602) q[2];
ry(0.1267306740647438) q[4];
cx q[2],q[4];
ry(0.4538740269670001) q[4];
ry(2.075895498590632) q[6];
cx q[4],q[6];
ry(1.128842750796941) q[4];
ry(2.1781656299946928) q[6];
cx q[4],q[6];
ry(2.257942212771656) q[1];
ry(2.0429197539586355) q[3];
cx q[1],q[3];
ry(-0.9102272321749421) q[1];
ry(-1.9025734342153848) q[3];
cx q[1],q[3];
ry(1.2670975782456448) q[3];
ry(-2.894132267176679) q[5];
cx q[3],q[5];
ry(-2.6538007317372547) q[3];
ry(-1.337455350012486) q[5];
cx q[3],q[5];
ry(1.2447022214427435) q[5];
ry(-0.39850509940217876) q[7];
cx q[5],q[7];
ry(-1.7598019897488806) q[5];
ry(-1.3853696473806407) q[7];
cx q[5],q[7];
ry(3.0969780471331623) q[0];
ry(-0.18182931248708378) q[3];
cx q[0],q[3];
ry(-1.8790792718960887) q[0];
ry(2.5266948815770816) q[3];
cx q[0],q[3];
ry(0.883640210574625) q[1];
ry(-1.103033296061799) q[2];
cx q[1],q[2];
ry(2.378892438506912) q[1];
ry(-3.019624766360323) q[2];
cx q[1],q[2];
ry(2.9447763569092364) q[2];
ry(1.2634490077267317) q[5];
cx q[2],q[5];
ry(1.8411419076498878) q[2];
ry(-1.061234985588829) q[5];
cx q[2],q[5];
ry(-2.759971333766276) q[3];
ry(1.5870143002647223) q[4];
cx q[3],q[4];
ry(2.1194158548777446) q[3];
ry(0.02940303668860711) q[4];
cx q[3],q[4];
ry(-0.4345914797407229) q[4];
ry(0.9287821259475573) q[7];
cx q[4],q[7];
ry(-0.32357132090285734) q[4];
ry(0.9931745875172755) q[7];
cx q[4],q[7];
ry(-2.3327141993341436) q[5];
ry(-1.6551095383566463) q[6];
cx q[5],q[6];
ry(-2.342357008780404) q[5];
ry(-1.6679527345557763) q[6];
cx q[5],q[6];
ry(-1.5940905285592075) q[0];
ry(-2.9173089760697417) q[1];
cx q[0],q[1];
ry(2.4449753218927035) q[0];
ry(-2.250963403111216) q[1];
cx q[0],q[1];
ry(-2.1216577063480324) q[2];
ry(1.603154990051441) q[3];
cx q[2],q[3];
ry(0.02739171091310915) q[2];
ry(-1.112564093583469) q[3];
cx q[2],q[3];
ry(2.719370731588313) q[4];
ry(-2.44915974745311) q[5];
cx q[4],q[5];
ry(-2.389418006157752) q[4];
ry(-2.399958103824334) q[5];
cx q[4],q[5];
ry(-3.070519552714879) q[6];
ry(-0.1717661875636669) q[7];
cx q[6],q[7];
ry(2.659693386957934) q[6];
ry(2.5259598305445947) q[7];
cx q[6],q[7];
ry(1.5814090560827214) q[0];
ry(1.448734709479969) q[2];
cx q[0],q[2];
ry(-0.041787041751234474) q[0];
ry(0.0762097752650654) q[2];
cx q[0],q[2];
ry(0.7308768339846125) q[2];
ry(1.0167903335347497) q[4];
cx q[2],q[4];
ry(-2.614220406005251) q[2];
ry(-2.790428450771514) q[4];
cx q[2],q[4];
ry(0.04277962897184553) q[4];
ry(2.3030897344668966) q[6];
cx q[4],q[6];
ry(-1.5674974310673477) q[4];
ry(-1.7056152725649) q[6];
cx q[4],q[6];
ry(1.809389625215565) q[1];
ry(-0.03782738011161779) q[3];
cx q[1],q[3];
ry(-3.136864243456023) q[1];
ry(-2.4495874585182604) q[3];
cx q[1],q[3];
ry(-1.2417937844554756) q[3];
ry(-0.1007041542336733) q[5];
cx q[3],q[5];
ry(-2.5757889434713963) q[3];
ry(-1.1915206663046325) q[5];
cx q[3],q[5];
ry(-1.200795834712558) q[5];
ry(0.5618401820660209) q[7];
cx q[5],q[7];
ry(1.9171698392923853) q[5];
ry(2.9244436950747272) q[7];
cx q[5],q[7];
ry(1.0250933700050502) q[0];
ry(0.5600932822385287) q[3];
cx q[0],q[3];
ry(-2.2160293725958082) q[0];
ry(-1.561892346166835) q[3];
cx q[0],q[3];
ry(-0.9550123298724422) q[1];
ry(1.8551063072418332) q[2];
cx q[1],q[2];
ry(-2.6934110718866067) q[1];
ry(-2.1810913219456762) q[2];
cx q[1],q[2];
ry(-3.081656284621193) q[2];
ry(2.1165115974251476) q[5];
cx q[2],q[5];
ry(2.883052367404568) q[2];
ry(-0.34245494981383917) q[5];
cx q[2],q[5];
ry(1.1245338583369335) q[3];
ry(-0.37998162618624853) q[4];
cx q[3],q[4];
ry(-0.5654255066721977) q[3];
ry(0.7484514106063154) q[4];
cx q[3],q[4];
ry(0.03444903795565022) q[4];
ry(0.4594711007686012) q[7];
cx q[4],q[7];
ry(2.946403725383195) q[4];
ry(-0.9891556096423448) q[7];
cx q[4],q[7];
ry(-2.9488597316065412) q[5];
ry(-0.5571483540435588) q[6];
cx q[5],q[6];
ry(2.4634876651816073) q[5];
ry(-2.299394468699634) q[6];
cx q[5],q[6];
ry(2.0851043312167103) q[0];
ry(2.5142394613610124) q[1];
cx q[0],q[1];
ry(2.4702349927214544) q[0];
ry(2.8793456857432855) q[1];
cx q[0],q[1];
ry(2.63366049937598) q[2];
ry(2.8176965106667327) q[3];
cx q[2],q[3];
ry(0.2770998528281608) q[2];
ry(-2.4413285925495996) q[3];
cx q[2],q[3];
ry(-2.010129943407632) q[4];
ry(-0.4391449189858631) q[5];
cx q[4],q[5];
ry(0.3801335237082988) q[4];
ry(-1.6722478946479082) q[5];
cx q[4],q[5];
ry(2.6122221745844962) q[6];
ry(-0.8523843509268559) q[7];
cx q[6],q[7];
ry(1.0228952235016793) q[6];
ry(-1.7900595812716313) q[7];
cx q[6],q[7];
ry(-1.6762032051135805) q[0];
ry(-2.3329519180478973) q[2];
cx q[0],q[2];
ry(1.3105105093244362) q[0];
ry(-2.042835513595081) q[2];
cx q[0],q[2];
ry(1.208720531477006) q[2];
ry(0.945007903153587) q[4];
cx q[2],q[4];
ry(1.803295229529927) q[2];
ry(-2.6462522629483765) q[4];
cx q[2],q[4];
ry(-1.7101444029953436) q[4];
ry(-0.2368147118529631) q[6];
cx q[4],q[6];
ry(-0.2439676870730878) q[4];
ry(-0.955854628306934) q[6];
cx q[4],q[6];
ry(2.2658967182219634) q[1];
ry(-2.55745978413253) q[3];
cx q[1],q[3];
ry(-2.9193948457163748) q[1];
ry(1.3462239713583175) q[3];
cx q[1],q[3];
ry(1.4865211004314927) q[3];
ry(2.376690700914963) q[5];
cx q[3],q[5];
ry(-1.7826828458249135) q[3];
ry(-0.5621211507966177) q[5];
cx q[3],q[5];
ry(0.06854209113030674) q[5];
ry(-2.927094680966271) q[7];
cx q[5],q[7];
ry(2.837386871823581) q[5];
ry(2.9628609060966697) q[7];
cx q[5],q[7];
ry(-0.11308420966670507) q[0];
ry(-0.0270750075335191) q[3];
cx q[0],q[3];
ry(1.8058333591331177) q[0];
ry(-3.0266070316684535) q[3];
cx q[0],q[3];
ry(-2.0336679522904246) q[1];
ry(1.7650867891541537) q[2];
cx q[1],q[2];
ry(2.212312060641924) q[1];
ry(0.8495391562498451) q[2];
cx q[1],q[2];
ry(-1.016251434427315) q[2];
ry(1.9960753310170796) q[5];
cx q[2],q[5];
ry(-2.737085127520377) q[2];
ry(2.611793531740079) q[5];
cx q[2],q[5];
ry(1.2041321681536872) q[3];
ry(-1.0138718826022979) q[4];
cx q[3],q[4];
ry(-1.0848361928000771) q[3];
ry(-1.8483015364501734) q[4];
cx q[3],q[4];
ry(-0.5010836130294605) q[4];
ry(-2.3561684297880707) q[7];
cx q[4],q[7];
ry(0.29727508152647086) q[4];
ry(-2.1829226204732537) q[7];
cx q[4],q[7];
ry(3.1214253093816295) q[5];
ry(-0.835968143133468) q[6];
cx q[5],q[6];
ry(-2.163951913046547) q[5];
ry(2.731322081334746) q[6];
cx q[5],q[6];
ry(-0.33735256877618014) q[0];
ry(1.3951655003933965) q[1];
cx q[0],q[1];
ry(1.6305257433908995) q[0];
ry(0.5481197890543967) q[1];
cx q[0],q[1];
ry(0.6320515298838517) q[2];
ry(1.158757855732385) q[3];
cx q[2],q[3];
ry(-1.630875781039817) q[2];
ry(-0.7718701440889948) q[3];
cx q[2],q[3];
ry(-2.2076946526675254) q[4];
ry(0.531449161028898) q[5];
cx q[4],q[5];
ry(-0.2594470979870209) q[4];
ry(0.6453081990583138) q[5];
cx q[4],q[5];
ry(2.2108987833542546) q[6];
ry(1.100289873127158) q[7];
cx q[6],q[7];
ry(0.9365458311208054) q[6];
ry(0.177202820454346) q[7];
cx q[6],q[7];
ry(-1.4534751146193683) q[0];
ry(-2.8768247011722425) q[2];
cx q[0],q[2];
ry(-1.554925715477835) q[0];
ry(-1.730377850912432) q[2];
cx q[0],q[2];
ry(0.6390540926710141) q[2];
ry(2.8450985790063297) q[4];
cx q[2],q[4];
ry(-2.945551401154775) q[2];
ry(-0.37157549508800525) q[4];
cx q[2],q[4];
ry(0.06931749713466129) q[4];
ry(1.5324846341743792) q[6];
cx q[4],q[6];
ry(-2.194489245524169) q[4];
ry(1.6866473786536307) q[6];
cx q[4],q[6];
ry(0.7801203204266091) q[1];
ry(2.2621896457908113) q[3];
cx q[1],q[3];
ry(2.0546535384060114) q[1];
ry(1.1090280618002772) q[3];
cx q[1],q[3];
ry(1.6932632866297699) q[3];
ry(0.2067657044987414) q[5];
cx q[3],q[5];
ry(2.6725642421689533) q[3];
ry(0.6403559423176302) q[5];
cx q[3],q[5];
ry(0.18792611095464995) q[5];
ry(-3.0651092693595694) q[7];
cx q[5],q[7];
ry(2.502077227297499) q[5];
ry(2.775091668103831) q[7];
cx q[5],q[7];
ry(-2.6556790046542873) q[0];
ry(-1.767390767744919) q[3];
cx q[0],q[3];
ry(-0.7859534224176109) q[0];
ry(1.8233510729576121) q[3];
cx q[0],q[3];
ry(2.0897936333600833) q[1];
ry(0.9889996400046404) q[2];
cx q[1],q[2];
ry(2.172893086701424) q[1];
ry(-1.1028631967022928) q[2];
cx q[1],q[2];
ry(-2.214272175957846) q[2];
ry(-1.3290893441493172) q[5];
cx q[2],q[5];
ry(1.6458882533602075) q[2];
ry(0.16744012376301054) q[5];
cx q[2],q[5];
ry(1.4425049304733795) q[3];
ry(0.9648964029134259) q[4];
cx q[3],q[4];
ry(0.2415225197158133) q[3];
ry(-2.570424131714964) q[4];
cx q[3],q[4];
ry(-0.22014499349020422) q[4];
ry(2.48171917810018) q[7];
cx q[4],q[7];
ry(-1.686403289829374) q[4];
ry(2.3158578958775675) q[7];
cx q[4],q[7];
ry(-1.0129569189754304) q[5];
ry(-0.5128379238175335) q[6];
cx q[5],q[6];
ry(-2.072793140119116) q[5];
ry(2.7282049015451717) q[6];
cx q[5],q[6];
ry(2.3795763934018934) q[0];
ry(-2.2221370575274775) q[1];
cx q[0],q[1];
ry(1.9141370258140828) q[0];
ry(1.1196166087791157) q[1];
cx q[0],q[1];
ry(1.570199164425186) q[2];
ry(-1.2447943063461597) q[3];
cx q[2],q[3];
ry(0.10334614436811584) q[2];
ry(0.9472411172700376) q[3];
cx q[2],q[3];
ry(2.82787947118796) q[4];
ry(-0.20938448134662835) q[5];
cx q[4],q[5];
ry(1.1164118153383547) q[4];
ry(0.008397223657590552) q[5];
cx q[4],q[5];
ry(1.4170137254449582) q[6];
ry(0.9201268989926451) q[7];
cx q[6],q[7];
ry(1.596910144774685) q[6];
ry(2.151476614648379) q[7];
cx q[6],q[7];
ry(-0.4131647710822028) q[0];
ry(0.08594991783662546) q[2];
cx q[0],q[2];
ry(-0.9554942401165607) q[0];
ry(-2.3883573346387976) q[2];
cx q[0],q[2];
ry(1.8556869615274971) q[2];
ry(2.0905279838041144) q[4];
cx q[2],q[4];
ry(-3.1364678649076714) q[2];
ry(1.9553127623901387) q[4];
cx q[2],q[4];
ry(1.3972994841890376) q[4];
ry(2.8275064043430533) q[6];
cx q[4],q[6];
ry(1.1440138384489345) q[4];
ry(2.569923111153406) q[6];
cx q[4],q[6];
ry(2.9870238838294676) q[1];
ry(-2.547381218635181) q[3];
cx q[1],q[3];
ry(2.13113297194482) q[1];
ry(-2.2382507358639208) q[3];
cx q[1],q[3];
ry(-2.9571870616218034) q[3];
ry(-0.9067398619236218) q[5];
cx q[3],q[5];
ry(0.953777329732942) q[3];
ry(-0.13906186713568333) q[5];
cx q[3],q[5];
ry(2.1158452072316756) q[5];
ry(-2.8081080299125296) q[7];
cx q[5],q[7];
ry(-0.8550600861507232) q[5];
ry(0.279150900762569) q[7];
cx q[5],q[7];
ry(-2.779987511416425) q[0];
ry(-1.1012154885729322) q[3];
cx q[0],q[3];
ry(-1.2937179466281221) q[0];
ry(0.24328481341240238) q[3];
cx q[0],q[3];
ry(0.13197223546230497) q[1];
ry(1.2336374573064957) q[2];
cx q[1],q[2];
ry(0.8807168746082503) q[1];
ry(-0.48771322582458865) q[2];
cx q[1],q[2];
ry(-2.9550439353518847) q[2];
ry(1.705176517363808) q[5];
cx q[2],q[5];
ry(0.10562060286707808) q[2];
ry(-2.409839230161748) q[5];
cx q[2],q[5];
ry(1.943561160235844) q[3];
ry(0.11586051459870905) q[4];
cx q[3],q[4];
ry(1.7656529684461495) q[3];
ry(1.0552400116809983) q[4];
cx q[3],q[4];
ry(-1.2989871094861163) q[4];
ry(-3.1078002169664325) q[7];
cx q[4],q[7];
ry(1.9023400240084678) q[4];
ry(1.3304464247738796) q[7];
cx q[4],q[7];
ry(0.9164663397668801) q[5];
ry(1.883597440685076) q[6];
cx q[5],q[6];
ry(-2.9835008296571286) q[5];
ry(0.03561764073237208) q[6];
cx q[5],q[6];
ry(2.632763948922979) q[0];
ry(-2.0889538248851016) q[1];
cx q[0],q[1];
ry(-1.972306741818131) q[0];
ry(0.37203183873269474) q[1];
cx q[0],q[1];
ry(1.9053221631293038) q[2];
ry(0.516219309604061) q[3];
cx q[2],q[3];
ry(0.9394068421439853) q[2];
ry(-0.12326440036745547) q[3];
cx q[2],q[3];
ry(2.96779575119263) q[4];
ry(0.9049239387704773) q[5];
cx q[4],q[5];
ry(2.425180536921967) q[4];
ry(-0.294214684166238) q[5];
cx q[4],q[5];
ry(-1.5685939709523933) q[6];
ry(1.2084517836177229) q[7];
cx q[6],q[7];
ry(-1.9073632533847231) q[6];
ry(2.180435838544763) q[7];
cx q[6],q[7];
ry(1.948458137419522) q[0];
ry(-1.0610897673342574) q[2];
cx q[0],q[2];
ry(2.473726809072275) q[0];
ry(2.867361424373866) q[2];
cx q[0],q[2];
ry(-2.4788985280821465) q[2];
ry(0.5662521203908986) q[4];
cx q[2],q[4];
ry(0.37137317434663475) q[2];
ry(-0.42627322191594) q[4];
cx q[2],q[4];
ry(-0.9287516559684685) q[4];
ry(1.9753978856456902) q[6];
cx q[4],q[6];
ry(-1.3709993007461176) q[4];
ry(-1.0746534170112028) q[6];
cx q[4],q[6];
ry(-2.123572620123963) q[1];
ry(0.693199979166596) q[3];
cx q[1],q[3];
ry(1.002130177411935) q[1];
ry(2.377949397758875) q[3];
cx q[1],q[3];
ry(1.7171016222224764) q[3];
ry(0.3434718782876729) q[5];
cx q[3],q[5];
ry(-3.039258482859518) q[3];
ry(-0.17762622089078217) q[5];
cx q[3],q[5];
ry(2.123766399234653) q[5];
ry(-1.7563636062531625) q[7];
cx q[5],q[7];
ry(2.0387261261001637) q[5];
ry(2.140950446349195) q[7];
cx q[5],q[7];
ry(-2.483616557344159) q[0];
ry(-0.870007640763995) q[3];
cx q[0],q[3];
ry(-0.5073218151287486) q[0];
ry(2.5025358586838093) q[3];
cx q[0],q[3];
ry(0.20644454754341623) q[1];
ry(-2.6783241146090524) q[2];
cx q[1],q[2];
ry(2.270331025219741) q[1];
ry(1.92016368124191) q[2];
cx q[1],q[2];
ry(2.9046211897751633) q[2];
ry(1.0938276658372317) q[5];
cx q[2],q[5];
ry(2.2451847086789507) q[2];
ry(1.052353769602587) q[5];
cx q[2],q[5];
ry(1.7657460544109327) q[3];
ry(-3.0779411015502136) q[4];
cx q[3],q[4];
ry(1.713031392951732) q[3];
ry(-2.228383291168317) q[4];
cx q[3],q[4];
ry(0.6413952870159106) q[4];
ry(-1.8086719248607483) q[7];
cx q[4],q[7];
ry(1.0251819256920358) q[4];
ry(-0.9427950233032316) q[7];
cx q[4],q[7];
ry(2.056827869931992) q[5];
ry(1.9932728903715873) q[6];
cx q[5],q[6];
ry(-0.9742384324276553) q[5];
ry(-0.9412552349042782) q[6];
cx q[5],q[6];
ry(-0.8082970136530196) q[0];
ry(-0.041183552751075325) q[1];
ry(2.91263425489849) q[2];
ry(1.3909788631704112) q[3];
ry(-1.8027121779205526) q[4];
ry(-1.915948925861867) q[5];
ry(-2.640079698898404) q[6];
ry(0.15845984337883845) q[7];