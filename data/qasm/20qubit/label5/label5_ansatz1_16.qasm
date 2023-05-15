OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.4481748646394979) q[0];
rz(-1.7708079898187803) q[0];
ry(0.15585321630097757) q[1];
rz(0.4230960336308556) q[1];
ry(2.4753582704505352) q[2];
rz(-0.3712781837365302) q[2];
ry(-0.7873680575629285) q[3];
rz(-1.5372673224323394) q[3];
ry(2.9424803216769) q[4];
rz(3.0920216700810963) q[4];
ry(0.7422342050719081) q[5];
rz(-2.340064144632398) q[5];
ry(-1.6832247685105912) q[6];
rz(1.2525675967270582) q[6];
ry(1.8874319138149127) q[7];
rz(1.350478115700605) q[7];
ry(3.138366341637602) q[8];
rz(-1.2712745427168295) q[8];
ry(-0.11023164510925643) q[9];
rz(-1.5722117147602628) q[9];
ry(0.03204071968503808) q[10];
rz(-2.042637826291224) q[10];
ry(0.9635140788485028) q[11];
rz(-0.17112449091565018) q[11];
ry(1.2542370258415156) q[12];
rz(-0.5877324549794882) q[12];
ry(1.24944012133862) q[13];
rz(-0.3758951806994464) q[13];
ry(0.008840087625212156) q[14];
rz(-0.18366709379266233) q[14];
ry(0.8608543395566234) q[15];
rz(-0.48192844097468784) q[15];
ry(0.029167677342800086) q[16];
rz(-1.128893888725433) q[16];
ry(-0.47240534870954715) q[17];
rz(-0.5489476138935212) q[17];
ry(-0.5076469475757537) q[18];
rz(-2.984356301020999) q[18];
ry(-2.109045544457169) q[19];
rz(2.4143450527069676) q[19];
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
ry(2.4809265933948654) q[0];
rz(-0.46360735699185257) q[0];
ry(-2.6217220417068288) q[1];
rz(0.14658893079855737) q[1];
ry(-0.3259068345788867) q[2];
rz(-1.6415054280086372) q[2];
ry(3.102108670026842) q[3];
rz(-2.6297772051464063) q[3];
ry(-3.1403959669127373) q[4];
rz(-1.437482972173485) q[4];
ry(3.072837291599244) q[5];
rz(-2.7757062756530053) q[5];
ry(0.00738999125132723) q[6];
rz(1.2162245692670814) q[6];
ry(-2.2070610283004823) q[7];
rz(-2.596785910436599) q[7];
ry(0.011197737167510269) q[8];
rz(0.7764360253132443) q[8];
ry(-0.003285471897577885) q[9];
rz(2.908971806663844) q[9];
ry(0.014747173860703273) q[10];
rz(1.3017680805178466) q[10];
ry(-0.08858310217663055) q[11];
rz(-2.297057714461715) q[11];
ry(1.2018648732345323) q[12];
rz(2.7160478926762455) q[12];
ry(2.722612695759666) q[13];
rz(2.5200236687792876) q[13];
ry(0.5189647653575449) q[14];
rz(-0.9329777210123941) q[14];
ry(-3.0659041202309893) q[15];
rz(2.839194907593006) q[15];
ry(0.00019949706223787445) q[16];
rz(-3.1048899479669134) q[16];
ry(-2.9340868713801185) q[17];
rz(0.06171603162701878) q[17];
ry(-2.0788422137828215) q[18];
rz(0.5067027029433183) q[18];
ry(-1.5665755748070547) q[19];
rz(0.6873924120476289) q[19];
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
ry(-3.1058376031579606) q[0];
rz(1.2122060093208216) q[0];
ry(3.1384398886952263) q[1];
rz(-0.3250779211460042) q[1];
ry(-3.1259988471740554) q[2];
rz(-1.744965199910963) q[2];
ry(-2.800057098827669) q[3];
rz(-0.08076139408245717) q[3];
ry(0.30573265089260193) q[4];
rz(-1.4415123791408568) q[4];
ry(2.331193093209329) q[5];
rz(-2.73516206271488) q[5];
ry(2.3810243963021134) q[6];
rz(1.83866072381209) q[6];
ry(-0.6103897545975876) q[7];
rz(2.2209836295945276) q[7];
ry(-3.131699523044576) q[8];
rz(1.4908711925575608) q[8];
ry(-1.8951956915336623) q[9];
rz(1.425699072541838) q[9];
ry(3.108398656330671) q[10];
rz(1.4558713407786872) q[10];
ry(2.2344288081141483) q[11];
rz(2.041764598802149) q[11];
ry(2.4969596313648332) q[12];
rz(0.44083049808329466) q[12];
ry(-0.07076844826940576) q[13];
rz(0.7458848731101685) q[13];
ry(0.2507493140761259) q[14];
rz(0.13069140148645975) q[14];
ry(0.7346702477229525) q[15];
rz(-1.9200811552232202) q[15];
ry(-3.1051194326439577) q[16];
rz(0.5069745288376293) q[16];
ry(0.06533673953974528) q[17];
rz(2.8886306147312943) q[17];
ry(-2.3338898838741056) q[18];
rz(0.6963358401506042) q[18];
ry(-0.22697121935961742) q[19];
rz(-0.92011615264233) q[19];
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
ry(3.131556080976207) q[0];
rz(1.734777344769249) q[0];
ry(2.1727331063328705) q[1];
rz(-0.793845336546271) q[1];
ry(-2.585073441889115) q[2];
rz(-1.8364288031299836) q[2];
ry(0.6460549731658167) q[3];
rz(1.5526848371298598) q[3];
ry(-3.1359646751804666) q[4];
rz(-1.819907477373808) q[4];
ry(-0.03861271986526607) q[5];
rz(1.1710308583735687) q[5];
ry(2.475550697040019) q[6];
rz(0.6468029489462621) q[6];
ry(0.41248612615544467) q[7];
rz(0.03597795817049842) q[7];
ry(-1.5743619519593324) q[8];
rz(-1.2431394075492668) q[8];
ry(1.892110545305442) q[9];
rz(-0.20907033331563785) q[9];
ry(-1.999815282686019) q[10];
rz(-1.3599535283732294) q[10];
ry(-0.9027995316957352) q[11];
rz(-1.088662569239452) q[11];
ry(0.5759142055118156) q[12];
rz(-1.046394948733722) q[12];
ry(-3.0499105131286925) q[13];
rz(-0.6677024371299041) q[13];
ry(-1.8889230456889106) q[14];
rz(-2.933363777452944) q[14];
ry(-3.1130208375665664) q[15];
rz(0.48132435682904884) q[15];
ry(-0.029468943110776635) q[16];
rz(-3.0319993390397824) q[16];
ry(-1.8186163222870666) q[17];
rz(-2.037680239450901) q[17];
ry(-0.14033171411182643) q[18];
rz(-2.623068352292279) q[18];
ry(1.9111088206022382) q[19];
rz(-0.40790654069122567) q[19];
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
ry(3.079429767246459) q[0];
rz(1.441732048377986) q[0];
ry(0.46197978147488605) q[1];
rz(1.3410818919366414) q[1];
ry(0.0630615230219725) q[2];
rz(-0.38291954535683814) q[2];
ry(-2.81345306672329) q[3];
rz(1.3134286044008805) q[3];
ry(2.399546385409845) q[4];
rz(-0.5058159778530787) q[4];
ry(-0.8013949315478244) q[5];
rz(-2.414232553242836) q[5];
ry(1.8115403388393094) q[6];
rz(-0.05373880022247623) q[6];
ry(-0.0013133109436198611) q[7];
rz(-1.2893118197054607) q[7];
ry(0.056068614960027574) q[8];
rz(-0.3116237574327388) q[8];
ry(-3.05492781897553) q[9];
rz(-2.5459577442859853) q[9];
ry(0.013840313792070041) q[10];
rz(1.741929025441929) q[10];
ry(2.9968942089277277) q[11];
rz(-0.09899830549761329) q[11];
ry(1.273951018971582) q[12];
rz(-2.540326452059411) q[12];
ry(-1.5765075554601533) q[13];
rz(-0.08901378393541616) q[13];
ry(0.2292333322309208) q[14];
rz(0.9001557022407846) q[14];
ry(1.3812404554129105) q[15];
rz(1.7785950808268414) q[15];
ry(-0.07565232960366863) q[16];
rz(-1.12557833303558) q[16];
ry(-0.7880318122966766) q[17];
rz(-2.312285639746387) q[17];
ry(-1.7678063320142137) q[18];
rz(2.7727990352751983) q[18];
ry(-2.6789496695377815) q[19];
rz(-1.9818925718338765) q[19];
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
ry(-0.7105949301886921) q[0];
rz(0.4424640279952801) q[0];
ry(-0.543445133374794) q[1];
rz(-0.16309633106908855) q[1];
ry(0.2506456488921884) q[2];
rz(-1.3726124360431253) q[2];
ry(-0.5733846293106886) q[3];
rz(0.5112905856187109) q[3];
ry(0.0008824669736160473) q[4];
rz(0.28845645161758693) q[4];
ry(-0.4484226685181009) q[5];
rz(-2.568360004271705) q[5];
ry(-2.20907745273947) q[6];
rz(-1.6537422385032843) q[6];
ry(-0.7381465505043013) q[7];
rz(-0.41567633378460567) q[7];
ry(0.03465550045416936) q[8];
rz(-0.05440868420707693) q[8];
ry(0.5153404107082331) q[9];
rz(1.3905508566347473) q[9];
ry(-2.6694631030016884) q[10];
rz(-0.3001734284981357) q[10];
ry(0.7294714547775216) q[11];
rz(0.07358349849670542) q[11];
ry(3.0689356175289886) q[12];
rz(1.2668164071865533) q[12];
ry(-0.5460293113810193) q[13];
rz(-1.2646253774030054) q[13];
ry(2.7491850866998777) q[14];
rz(-1.9735197172319703) q[14];
ry(-0.2260089571776241) q[15];
rz(1.7702657135902358) q[15];
ry(2.7701307529796355) q[16];
rz(-1.1438792040452563) q[16];
ry(-3.116074230151215) q[17];
rz(1.7828922597244772) q[17];
ry(-2.4113643478231466) q[18];
rz(-1.1951983219211992) q[18];
ry(2.3126841346542526) q[19];
rz(-0.44658251418998507) q[19];
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
ry(2.222201296307446) q[0];
rz(2.266729438147231) q[0];
ry(-1.4585745956279244) q[1];
rz(-1.3031030452172456) q[1];
ry(1.9772781569237985) q[2];
rz(2.857971382583817) q[2];
ry(0.09262384343029592) q[3];
rz(-1.5018392112501264) q[3];
ry(-1.1564994141732372) q[4];
rz(-0.040578226538281414) q[4];
ry(2.117771268553926) q[5];
rz(0.3219920753034531) q[5];
ry(0.023146486672555255) q[6];
rz(-0.44941942802578433) q[6];
ry(-3.1413291933922656) q[7];
rz(-2.977244868281427) q[7];
ry(2.012658000906784) q[8];
rz(-2.74927254887709) q[8];
ry(-3.0860785835010303) q[9];
rz(-1.6266934581160404) q[9];
ry(-3.1380444730963557) q[10];
rz(-2.0771635532733863) q[10];
ry(-1.2871798025371668) q[11];
rz(0.024156526380389155) q[11];
ry(-2.9242329260092603) q[12];
rz(-2.077906051374227) q[12];
ry(-0.04197859996064324) q[13];
rz(1.375306649607151) q[13];
ry(-3.100596573137788) q[14];
rz(-1.3780105566372143) q[14];
ry(-2.9045510910670496) q[15];
rz(-0.3226683501205726) q[15];
ry(-0.15155068385074774) q[16];
rz(-2.073186668347889) q[16];
ry(-0.20143695610429566) q[17];
rz(0.20116258535562362) q[17];
ry(-2.3548123262585254) q[18];
rz(1.4242965543927422) q[18];
ry(-1.784217600964521) q[19];
rz(-0.33788140102023956) q[19];
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
ry(0.01675822555601598) q[0];
rz(0.08678877917474459) q[0];
ry(1.7990981308145302) q[1];
rz(0.518837457621367) q[1];
ry(1.5389778998658092) q[2];
rz(1.5540771786100622) q[2];
ry(1.5796099005129296) q[3];
rz(0.1269212040010844) q[3];
ry(3.134717398178037) q[4];
rz(-1.5849366258089876) q[4];
ry(-1.7401370172823556) q[5];
rz(1.570092884713844) q[5];
ry(-1.1876370225048354) q[6];
rz(-1.2301145540342622) q[6];
ry(0.6645372671814848) q[7];
rz(-2.7141263282366337) q[7];
ry(-0.0395522540638904) q[8];
rz(-0.37777484753802026) q[8];
ry(-2.0582938698533884) q[9];
rz(-0.5899695118883068) q[9];
ry(-0.008290026839046359) q[10];
rz(2.8787673472153146) q[10];
ry(-0.8280455362291896) q[11];
rz(-3.1395189063706703) q[11];
ry(-0.627384774211733) q[12];
rz(-1.2156034571530876) q[12];
ry(-0.19834654306795432) q[13];
rz(3.0893413435253576) q[13];
ry(3.1352379336708434) q[14];
rz(1.6642692476853274) q[14];
ry(-0.12084707449179519) q[15];
rz(-2.696072684504991) q[15];
ry(0.37656754739315695) q[16];
rz(-1.0287518871649803) q[16];
ry(-0.010227033147867992) q[17];
rz(2.207367028094631) q[17];
ry(-2.0662686934445693) q[18];
rz(1.2494875726287713) q[18];
ry(0.24744268420931748) q[19];
rz(-0.10155451288521829) q[19];
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
ry(0.16537333126048726) q[0];
rz(2.8995950853378836) q[0];
ry(-1.3881493515953585) q[1];
rz(-2.04414277999892) q[1];
ry(-1.583969620810209) q[2];
rz(-3.0971418080754374) q[2];
ry(-0.2361761659067323) q[3];
rz(-1.6678321360020725) q[3];
ry(1.8402428001840718) q[4];
rz(-0.0020002484869157426) q[4];
ry(1.533368832594399) q[5];
rz(-0.626423579448873) q[5];
ry(0.0005669324290538835) q[6];
rz(-1.5347167829140669) q[6];
ry(-3.1403603770995026) q[7];
rz(-2.6967625507690407) q[7];
ry(-2.347819339688499) q[8];
rz(-3.1413610923244493) q[8];
ry(-1.0322028512552164) q[9];
rz(1.9277643577058692) q[9];
ry(2.65337848251751) q[10];
rz(2.9823049502823644) q[10];
ry(1.765171871288313) q[11];
rz(-2.2297641582873684) q[11];
ry(1.0529577456413661) q[12];
rz(2.4920815700983927) q[12];
ry(1.5879575558830747) q[13];
rz(0.39610287282531365) q[13];
ry(-3.087774983441861) q[14];
rz(1.0524798500559183) q[14];
ry(1.2640866423394204) q[15];
rz(-0.5111548269664263) q[15];
ry(-0.060176105388009216) q[16];
rz(-0.3541005904319876) q[16];
ry(1.679318294370149) q[17];
rz(0.5419287463214086) q[17];
ry(0.33228683014165217) q[18];
rz(-0.2533971839399472) q[18];
ry(2.3771190790593915) q[19];
rz(2.9082850947483125) q[19];
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
ry(-0.005065028533072002) q[0];
rz(2.9011590733607915) q[0];
ry(-3.0348959060627516) q[1];
rz(-0.464042103914907) q[1];
ry(1.6312792187779466) q[2];
rz(-1.9062293258192424) q[2];
ry(0.001285480016403051) q[3];
rz(0.6029976687999009) q[3];
ry(-2.2962324575507673) q[4];
rz(-3.1167138858674504) q[4];
ry(-1.709335254405552) q[5];
rz(-1.9791963576116274) q[5];
ry(1.6501861407903498) q[6];
rz(-1.542897060694604) q[6];
ry(1.0011540162547918) q[7];
rz(0.010626349340762786) q[7];
ry(-1.9175341781621953) q[8];
rz(0.004194309189601356) q[8];
ry(-3.1220657185518603) q[9];
rz(1.8900830818013403) q[9];
ry(0.08104643879256201) q[10];
rz(1.580382376258886) q[10];
ry(0.037314487478407266) q[11];
rz(-2.496207433433061) q[11];
ry(-1.5606650345099728) q[12];
rz(-0.5275611768535796) q[12];
ry(3.128935934193852) q[13];
rz(-1.6595654714463177) q[13];
ry(-1.8849096940455068) q[14];
rz(-3.1280472288431946) q[14];
ry(1.3283426355629064) q[15];
rz(-0.26036082139116173) q[15];
ry(-1.5766506889903322) q[16];
rz(-1.560971930476839) q[16];
ry(1.9783171795826728) q[17];
rz(-2.819079721248736) q[17];
ry(2.6285852728130106) q[18];
rz(2.932361606955679) q[18];
ry(-2.329459987853358) q[19];
rz(2.216364043582006) q[19];
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
ry(-2.9144739615030044) q[0];
rz(3.0838998162368347) q[0];
ry(1.4222531061744548) q[1];
rz(0.8106719335303757) q[1];
ry(-0.06597793968090385) q[2];
rz(2.8512709355139383) q[2];
ry(0.026726538776521203) q[3];
rz(-0.40369055759995226) q[3];
ry(1.5740035889200268) q[4];
rz(-0.689675609084948) q[4];
ry(-3.0883946765821726) q[5];
rz(1.0578402026558933) q[5];
ry(-2.070023866470608) q[6];
rz(-0.00010660055837477654) q[6];
ry(-2.070834984007834) q[7];
rz(-0.00895707164663358) q[7];
ry(-0.7479910550939888) q[8];
rz(3.1322442013370795) q[8];
ry(1.6285533719251921) q[9];
rz(-1.4737145588746983) q[9];
ry(0.45047525226616886) q[10];
rz(-3.1086880752920427) q[10];
ry(1.5100444264824777) q[11];
rz(-2.624194337379613) q[11];
ry(0.6959915033685364) q[12];
rz(-1.1189682968091714) q[12];
ry(-3.1052245227567807) q[13];
rz(2.7166117097545994) q[13];
ry(-1.6689825801518579) q[14];
rz(-0.007243063302283993) q[14];
ry(1.2155578259430566) q[15];
rz(0.002065759897241115) q[15];
ry(-0.7183675156261355) q[16];
rz(-3.1359884265684146) q[16];
ry(2.360679381498786) q[17];
rz(1.6241517890469437) q[17];
ry(2.171867150085247) q[18];
rz(-0.986524095763964) q[18];
ry(1.5214656212218547) q[19];
rz(-0.4658882071366355) q[19];
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
ry(-1.595900782636936) q[0];
rz(-3.04076872646406) q[0];
ry(1.0069551103011696) q[1];
rz(-1.6129078218236437) q[1];
ry(2.3802237641706685) q[2];
rz(2.354752490717655) q[2];
ry(3.127975717140765) q[3];
rz(-2.9298859587704458) q[3];
ry(-0.008467222495506776) q[4];
rz(1.541773441675752) q[4];
ry(0.15947634896907553) q[5];
rz(0.12231644266883547) q[5];
ry(-1.7384878165994615) q[6];
rz(-0.7874570451722134) q[6];
ry(-1.4731066961250614) q[7];
rz(-0.009000240113830318) q[7];
ry(-1.8149183315542583) q[8];
rz(3.1401458444952803) q[8];
ry(-3.1286798723097156) q[9];
rz(1.67601486557618) q[9];
ry(9.556673300892271e-05) q[10];
rz(1.743898779917691) q[10];
ry(3.1253743771675584) q[11];
rz(0.48249080764741503) q[11];
ry(2.6123362157559127) q[12];
rz(-0.039150217168874235) q[12];
ry(-0.16337566999420006) q[13];
rz(3.090799344787519) q[13];
ry(-2.797496239462726) q[14];
rz(0.42453535240735174) q[14];
ry(-1.051388491670678) q[15];
rz(3.1228261122489225) q[15];
ry(-1.0840783725992393) q[16];
rz(-0.4001983914403041) q[16];
ry(3.1333254866746887) q[17];
rz(0.542962562215223) q[17];
ry(-3.06881498703114) q[18];
rz(0.8571477774722586) q[18];
ry(3.0624456890358602) q[19];
rz(1.3577663061623488) q[19];
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
ry(-3.0803522181460155) q[0];
rz(2.406194451911741) q[0];
ry(1.390914150296748) q[1];
rz(-3.011723740288405) q[1];
ry(-1.5680367436785208) q[2];
rz(-3.134274265393014) q[2];
ry(-1.4884278102306945) q[3];
rz(0.054793888558436504) q[3];
ry(-3.1090273717280574) q[4];
rz(-2.292545130591666) q[4];
ry(1.136994357569483) q[5];
rz(0.7026800549962484) q[5];
ry(-3.1340792929888877) q[6];
rz(2.965539435896148) q[6];
ry(-1.6803065607940137) q[7];
rz(2.9118045899951284) q[7];
ry(-1.9790996204134306) q[8];
rz(3.1397029007890636) q[8];
ry(-2.1902775954487392) q[9];
rz(-3.1001047726748427) q[9];
ry(1.9283748005571741) q[10];
rz(-2.712730257032698) q[10];
ry(2.1262402130764677) q[11];
rz(3.0954047728358827) q[11];
ry(0.6207849085340458) q[12];
rz(-0.05329050618271314) q[12];
ry(-2.5709773114570633) q[13];
rz(2.7133210832803956) q[13];
ry(-2.572260464322704) q[14];
rz(0.3531226666446198) q[14];
ry(1.7669786501604836) q[15];
rz(0.9468851540482756) q[15];
ry(0.4049639113278309) q[16];
rz(2.9441988822641987) q[16];
ry(-2.200270589913986) q[17];
rz(2.682007377205997) q[17];
ry(1.5973882390558818) q[18];
rz(-0.062236124600635205) q[18];
ry(2.878263891750032) q[19];
rz(3.1348753656972335) q[19];
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
ry(-2.141255754057643) q[0];
rz(-1.418383966193778) q[0];
ry(0.06810408816841253) q[1];
rz(-0.706636621027355) q[1];
ry(-2.2619272973207236) q[2];
rz(-3.1390239973614955) q[2];
ry(-1.800689782764791) q[3];
rz(-3.104085124927987) q[3];
ry(1.435910178492744) q[4];
rz(-3.138979755115545) q[4];
ry(-0.005903649370332734) q[5];
rz(-0.7852941345168921) q[5];
ry(-3.1394776927666066) q[6];
rz(0.61130503870057) q[6];
ry(-0.006573770177389967) q[7];
rz(-2.9196813156992216) q[7];
ry(-1.7119461980361654) q[8];
rz(0.00029664934147762196) q[8];
ry(1.3076978552065048) q[9];
rz(-0.0024643572526432678) q[9];
ry(0.6777664835692699) q[10];
rz(3.1412611368954133) q[10];
ry(2.5561993190623826) q[11];
rz(3.112746900925336) q[11];
ry(0.2042422174318883) q[12];
rz(-2.9166175905730447) q[12];
ry(0.017945706790684208) q[13];
rz(2.674618445818827) q[13];
ry(2.6466817022578586) q[14];
rz(3.133968407046705) q[14];
ry(0.015890596018129706) q[15];
rz(2.2288014746715366) q[15];
ry(0.018695763044475733) q[16];
rz(-2.591244917705713) q[16];
ry(-3.138060279448742) q[17];
rz(-1.9868913081921185) q[17];
ry(3.014906306250651) q[18];
rz(-1.5124440672772614) q[18];
ry(-0.23874097627372937) q[19];
rz(0.4804870981119125) q[19];
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
ry(-3.1154427379603886) q[0];
rz(1.3827854753915803) q[0];
ry(0.011897663919473443) q[1];
rz(-1.5131736555558788) q[1];
ry(1.4185104301364808) q[2];
rz(-3.1260546762551464) q[2];
ry(2.285928338329079) q[3];
rz(-3.029459373559486) q[3];
ry(1.3555698988044422) q[4];
rz(3.089272876634552) q[4];
ry(-3.13215713379235) q[5];
rz(3.0446414606642414) q[5];
ry(-1.7015709124368437) q[6];
rz(-0.00411709417607634) q[6];
ry(-2.483675038958143) q[7];
rz(3.1348965035170515) q[7];
ry(0.928293034435769) q[8];
rz(3.1405085776679087) q[8];
ry(-1.4646674647020423) q[9];
rz(2.4838078567354964) q[9];
ry(1.460962200638063) q[10];
rz(3.1396583960283655) q[10];
ry(2.0399406876811574) q[11];
rz(-3.1274367015549207) q[11];
ry(-2.877222843190279) q[12];
rz(-3.032094054882397) q[12];
ry(3.1388392907378693) q[13];
rz(-0.8636205757328446) q[13];
ry(-1.5782352925609366) q[14];
rz(-2.292395191715066) q[14];
ry(3.0785027909838387) q[15];
rz(-3.1206291384944223) q[15];
ry(1.441468525530031) q[16];
rz(0.23278071232617936) q[16];
ry(-0.9682151066574193) q[17];
rz(-1.0495456183457126) q[17];
ry(1.7361408642409706) q[18];
rz(-2.0607853768158604) q[18];
ry(0.18526230927616258) q[19];
rz(-0.8137383235233829) q[19];
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
ry(2.4019733145597817) q[0];
rz(2.2031714308320502) q[0];
ry(-3.024192447501513) q[1];
rz(1.053133951386438) q[1];
ry(-0.06969862475957317) q[2];
rz(2.7655652524228804) q[2];
ry(-2.8203848833242415) q[3];
rz(2.5082136156908224) q[3];
ry(3.1289576553193643) q[4];
rz(3.0904216091187386) q[4];
ry(1.642328651549839) q[5];
rz(-0.003472686993514241) q[5];
ry(2.112243467318902) q[6];
rz(0.0008931669937629682) q[6];
ry(1.5182729295918598) q[7];
rz(-3.1330565826696315) q[7];
ry(-1.0190167539399422) q[8];
rz(3.1378456246323565) q[8];
ry(-0.017331772178693864) q[9];
rz(-1.6074337487665753) q[9];
ry(-2.4173388152088373) q[10];
rz(3.138918529190656) q[10];
ry(2.599850949899681) q[11];
rz(-3.1287193060017655) q[11];
ry(-1.8178919348369762) q[12];
rz(3.133849266362243) q[12];
ry(-1.3275136289184104) q[13];
rz(3.133967459232166) q[13];
ry(2.9805536668680483) q[14];
rz(0.6635698562671566) q[14];
ry(-1.339809660352004) q[15];
rz(-3.1223552401894743) q[15];
ry(0.2628328336779738) q[16];
rz(3.0412793727271707) q[16];
ry(1.0839263337942588) q[17];
rz(3.1375947910679756) q[17];
ry(-3.11402853773613) q[18];
rz(0.8896790359360551) q[18];
ry(1.8203566829088516) q[19];
rz(3.10901150224696) q[19];
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
ry(-2.51012036321792) q[0];
rz(0.014426145394352913) q[0];
ry(-1.113532482234373) q[1];
rz(-0.6031767404063695) q[1];
ry(3.139301489544146) q[2];
rz(2.7811865961400746) q[2];
ry(-3.10337705056358) q[3];
rz(-0.7213762251421859) q[3];
ry(2.561373867499643) q[4];
rz(0.0002939215141175566) q[4];
ry(-1.5767335174160666) q[5];
rz(0.0025652454929749875) q[5];
ry(1.7220192576666458) q[6];
rz(-3.140801134365795) q[6];
ry(0.05282072305141256) q[7];
rz(3.132786009720274) q[7];
ry(0.6355247141232944) q[8];
rz(3.1415445090238916) q[8];
ry(-3.1362440401559137) q[9];
rz(-2.269156853678683) q[9];
ry(2.9557027436305514) q[10];
rz(-0.0346732604208011) q[10];
ry(-0.8332696891144346) q[11];
rz(-0.010878324888377094) q[11];
ry(-1.2723360393132404) q[12];
rz(0.002509731609508847) q[12];
ry(2.4057486525381577) q[13];
rz(-3.136123800738788) q[13];
ry(0.5145221086922085) q[14];
rz(3.1391143099234298) q[14];
ry(2.9949647393906424) q[15];
rz(0.014241879195349139) q[15];
ry(2.7641523698338966) q[16];
rz(-2.557362411385247) q[16];
ry(-2.5968257741494036) q[17];
rz(-1.6217236481577992) q[17];
ry(-1.7904699824839732) q[18];
rz(2.437261141424) q[18];
ry(-1.6882873280413921) q[19];
rz(-0.11538357526423812) q[19];
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
ry(-3.136670384949561) q[0];
rz(0.8194327562358868) q[0];
ry(3.130116602668273) q[1];
rz(2.5366531628161506) q[1];
ry(1.864554097471261) q[2];
rz(-1.765603734429603) q[2];
ry(1.7630551723295085) q[3];
rz(-0.016824073874968985) q[3];
ry(-1.6550065402903584) q[4];
rz(-0.003719514117726419) q[4];
ry(-1.9379235042624057) q[5];
rz(3.1410575415992175) q[5];
ry(1.588397136246308) q[6];
rz(-1.1276338328566489) q[6];
ry(0.2374689559254213) q[7];
rz(0.02836240633955889) q[7];
ry(0.014289072481812326) q[8];
rz(-3.14044142638644) q[8];
ry(2.3266496651291204) q[9];
rz(0.006831633533035897) q[9];
ry(3.080673892392269) q[10];
rz(-0.033495244445679084) q[10];
ry(-1.6586136780172271) q[11];
rz(-3.137473743782569) q[11];
ry(1.8125193771610242) q[12];
rz(0.005309787789020248) q[12];
ry(-0.9954311909953873) q[13];
rz(-3.140748461464531) q[13];
ry(1.3999660015496929) q[14];
rz(-3.1377101381768506) q[14];
ry(-1.9818565407192892) q[15];
rz(1.2868281850200824) q[15];
ry(-0.30292616860151705) q[16];
rz(-0.32707156012680283) q[16];
ry(-1.8680420333765069) q[17];
rz(2.2478180699356916) q[17];
ry(-1.8442349368512696) q[18];
rz(-1.0398650840888548) q[18];
ry(1.1441499806718642) q[19];
rz(0.25160506614411626) q[19];
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
ry(-1.1277697616585678) q[0];
rz(-2.005797540722827) q[0];
ry(-2.1788352375407074) q[1];
rz(-1.6436027950549457) q[1];
ry(3.1394665250408336) q[2];
rz(-0.19663945162645433) q[2];
ry(0.7000213944931359) q[3];
rz(-1.5791169945048573) q[3];
ry(0.7199595506874319) q[4];
rz(-1.5685789828217098) q[4];
ry(1.6012650538604065) q[5];
rz(1.5728700712487649) q[5];
ry(-0.00034046052259199355) q[6];
rz(2.6985743221373415) q[6];
ry(0.0003078650460812682) q[7];
rz(-1.6011835399473535) q[7];
ry(0.9243493406219462) q[8];
rz(-1.5683599428673318) q[8];
ry(2.2015975356721) q[9];
rz(-1.5694150544368748) q[9];
ry(-1.248600331775759) q[10];
rz(-1.5721472354918538) q[10];
ry(-0.6848140295894606) q[11];
rz(1.5598373502029284) q[11];
ry(0.2113520378174929) q[12];
rz(1.5660139393469923) q[12];
ry(-1.0529130848825874) q[13];
rz(-1.5695978337520529) q[13];
ry(-2.226963900073251) q[14];
rz(1.5694888165693932) q[14];
ry(3.1326555673355574) q[15];
rz(2.8443961258571515) q[15];
ry(3.140207426046388) q[16];
rz(1.8290798158378072) q[16];
ry(3.1349371661418353) q[17];
rz(-0.7386057950656556) q[17];
ry(0.007054921140575488) q[18];
rz(2.386986754900293) q[18];
ry(-0.015116614095955863) q[19];
rz(-0.35971510982865595) q[19];
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
ry(1.5808017457156565) q[0];
rz(-1.9698753107407851) q[0];
ry(1.5845907696555912) q[1];
rz(2.6830867088195203) q[1];
ry(1.5733560744328292) q[2];
rz(-3.0978369396749788) q[2];
ry(1.566879318256662) q[3];
rz(-0.6771419267120427) q[3];
ry(-1.570679345174297) q[4];
rz(-2.2125910062305625) q[4];
ry(-1.5732108510736502) q[5];
rz(-1.834705820484169) q[5];
ry(1.571380549330875) q[6];
rz(2.5112804728642475) q[6];
ry(-1.5684199364836988) q[7];
rz(-0.9533230539064553) q[7];
ry(1.5704387131327957) q[8];
rz(-2.7504236536752096) q[8];
ry(1.5745697868710578) q[9];
rz(0.7481973577680243) q[9];
ry(-1.5710208687231235) q[10];
rz(-0.36945966345080805) q[10];
ry(-1.572451931754106) q[11];
rz(-0.14199925428934626) q[11];
ry(-1.5718252516764952) q[12];
rz(0.046578702547719515) q[12];
ry(-1.5743072884332856) q[13];
rz(-1.1839300148842735) q[13];
ry(-1.5707649588820956) q[14];
rz(-1.407370476245732) q[14];
ry(1.5663151333271248) q[15];
rz(-2.334675218464938) q[15];
ry(-1.7655480471233158) q[16];
rz(-0.06665707504331153) q[16];
ry(0.2972028984468119) q[17];
rz(-2.7001380853836707) q[17];
ry(-0.9018995405566066) q[18];
rz(3.0438960416732326) q[18];
ry(-1.3873757613818354) q[19];
rz(-0.6147812502917215) q[19];