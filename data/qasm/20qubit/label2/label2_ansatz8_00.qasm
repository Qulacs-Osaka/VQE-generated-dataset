OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.3686425220080825) q[0];
ry(2.0592864031275075) q[1];
cx q[0],q[1];
ry(-2.9849824916832874) q[0];
ry(-2.3661743329984297) q[1];
cx q[0],q[1];
ry(-2.8555449556004318) q[2];
ry(-0.08621625940611999) q[3];
cx q[2],q[3];
ry(-2.480320085086652) q[2];
ry(-2.039331967040229) q[3];
cx q[2],q[3];
ry(3.1173553437261012) q[4];
ry(1.0815057593401871) q[5];
cx q[4],q[5];
ry(-2.744769552537049) q[4];
ry(-0.5176403714871967) q[5];
cx q[4],q[5];
ry(-2.667423904824419) q[6];
ry(3.1251976200955065) q[7];
cx q[6],q[7];
ry(-2.7896036690539456) q[6];
ry(-2.4680842127571188) q[7];
cx q[6],q[7];
ry(-1.04682174064672) q[8];
ry(-3.017154662187996) q[9];
cx q[8],q[9];
ry(0.0917256990704902) q[8];
ry(0.129642320930833) q[9];
cx q[8],q[9];
ry(2.478021941504779) q[10];
ry(-2.192329505730985) q[11];
cx q[10],q[11];
ry(-0.09275079871254444) q[10];
ry(0.08382766779556938) q[11];
cx q[10],q[11];
ry(1.3239646903405964) q[12];
ry(-1.9120279984443471) q[13];
cx q[12],q[13];
ry(-2.3465041937739093) q[12];
ry(2.821202687524717) q[13];
cx q[12],q[13];
ry(1.684942610302184) q[14];
ry(0.0002788030697349342) q[15];
cx q[14],q[15];
ry(-1.637012868387398) q[14];
ry(-0.06708837680591095) q[15];
cx q[14],q[15];
ry(2.1296064026816337) q[16];
ry(2.9883575682540884) q[17];
cx q[16],q[17];
ry(1.365866223410303) q[16];
ry(1.6721084954225756) q[17];
cx q[16],q[17];
ry(1.1136230123318303) q[18];
ry(-0.6325787112173433) q[19];
cx q[18],q[19];
ry(-2.355897549797787) q[18];
ry(-2.2261362703537415) q[19];
cx q[18],q[19];
ry(1.3733264872160253) q[0];
ry(2.043214150637488) q[2];
cx q[0],q[2];
ry(3.0624946170024105) q[0];
ry(3.0380011864186613) q[2];
cx q[0],q[2];
ry(1.509946350229293) q[2];
ry(2.0950052133204835) q[4];
cx q[2],q[4];
ry(0.014479635372445012) q[2];
ry(0.018361762634079605) q[4];
cx q[2],q[4];
ry(-2.099640805601882) q[4];
ry(-0.9932549756146124) q[6];
cx q[4],q[6];
ry(-1.4610545962533843) q[4];
ry(0.013585096455150492) q[6];
cx q[4],q[6];
ry(-1.9161934909424418) q[6];
ry(0.10440043054901696) q[8];
cx q[6],q[8];
ry(-1.4705529749672772) q[6];
ry(0.7097816688254586) q[8];
cx q[6],q[8];
ry(-2.35231080869528) q[8];
ry(-1.02508871522202) q[10];
cx q[8],q[10];
ry(-2.6596707247127163) q[8];
ry(0.013793644051586) q[10];
cx q[8],q[10];
ry(-1.4490475628579533) q[10];
ry(-1.2358263140306776) q[12];
cx q[10],q[12];
ry(-1.9349396534189476) q[10];
ry(0.21978293398904383) q[12];
cx q[10],q[12];
ry(-1.7475439739571974) q[12];
ry(-1.4543893173853149) q[14];
cx q[12],q[14];
ry(2.973894146193163) q[12];
ry(2.1407458449407417) q[14];
cx q[12],q[14];
ry(2.4591035470212557) q[14];
ry(-2.8907804140237734) q[16];
cx q[14],q[16];
ry(2.665573801889897) q[14];
ry(0.008149362606773994) q[16];
cx q[14],q[16];
ry(2.565929664849941) q[16];
ry(-0.9273193157671872) q[18];
cx q[16],q[18];
ry(-1.7412988140509087) q[16];
ry(2.872629429121507) q[18];
cx q[16],q[18];
ry(-1.8345017390578213) q[1];
ry(-2.823070091561236) q[3];
cx q[1],q[3];
ry(1.893934686176032) q[1];
ry(-2.432797588970191) q[3];
cx q[1],q[3];
ry(2.995051857479852) q[3];
ry(-3.13292638266162) q[5];
cx q[3],q[5];
ry(0.45861646504720505) q[3];
ry(3.125046766667893) q[5];
cx q[3],q[5];
ry(0.24809637779061014) q[5];
ry(1.2532112894747809) q[7];
cx q[5],q[7];
ry(-0.4244942510982215) q[5];
ry(-0.21368667853682988) q[7];
cx q[5],q[7];
ry(2.789378913067475) q[7];
ry(1.3202291195391727) q[9];
cx q[7],q[9];
ry(0.1128280947532323) q[7];
ry(0.06885602323141705) q[9];
cx q[7],q[9];
ry(2.695431164680152) q[9];
ry(-0.9388134214103241) q[11];
cx q[9],q[11];
ry(-3.120786870233493) q[9];
ry(-0.027266463712489114) q[11];
cx q[9],q[11];
ry(-1.8809348031363449) q[11];
ry(-1.2473104036863578) q[13];
cx q[11],q[13];
ry(-2.2167582760977935) q[11];
ry(2.4539400122376143) q[13];
cx q[11],q[13];
ry(-0.06830080885163525) q[13];
ry(-0.8389117039906833) q[15];
cx q[13],q[15];
ry(1.6084322511332267) q[13];
ry(-0.9262811711700047) q[15];
cx q[13],q[15];
ry(-2.093405340316929) q[15];
ry(-2.120990838290843) q[17];
cx q[15],q[17];
ry(-0.5056039742253874) q[15];
ry(-0.014083621949598779) q[17];
cx q[15],q[17];
ry(-0.10585293074720634) q[17];
ry(1.796051379562007) q[19];
cx q[17],q[19];
ry(-2.8590091123534886) q[17];
ry(0.03933265227948586) q[19];
cx q[17],q[19];
ry(0.9410137693839671) q[0];
ry(-2.045913919357384) q[1];
cx q[0],q[1];
ry(3.07560521903932) q[0];
ry(1.6451904451338946) q[1];
cx q[0],q[1];
ry(-1.0167551917631732) q[2];
ry(-2.9537798720521957) q[3];
cx q[2],q[3];
ry(-1.3579150651056624) q[2];
ry(-1.2044959822343262) q[3];
cx q[2],q[3];
ry(-2.2214366591732206) q[4];
ry(-0.48511022041699847) q[5];
cx q[4],q[5];
ry(1.8847945356406814) q[4];
ry(1.0142151373524964) q[5];
cx q[4],q[5];
ry(2.854568368987669) q[6];
ry(-2.129450091645012) q[7];
cx q[6],q[7];
ry(1.0395946171130266) q[6];
ry(1.662498203538129) q[7];
cx q[6],q[7];
ry(-1.4620644651325962) q[8];
ry(-0.16349450197934087) q[9];
cx q[8],q[9];
ry(2.3378988911563243) q[8];
ry(-1.2984639320680518) q[9];
cx q[8],q[9];
ry(-2.5743028989166477) q[10];
ry(-0.03825743185131536) q[11];
cx q[10],q[11];
ry(2.6614337470671403) q[10];
ry(-0.927175713170211) q[11];
cx q[10],q[11];
ry(-0.14984498080267308) q[12];
ry(2.4483610520511587) q[13];
cx q[12],q[13];
ry(-0.10245723369585225) q[12];
ry(-3.1314629053433065) q[13];
cx q[12],q[13];
ry(-1.0790951297145837) q[14];
ry(0.09857495600627268) q[15];
cx q[14],q[15];
ry(-3.0752730841477787) q[14];
ry(3.013076179338632) q[15];
cx q[14],q[15];
ry(0.7548931923034053) q[16];
ry(-0.7309949709623023) q[17];
cx q[16],q[17];
ry(-0.32165695540674294) q[16];
ry(-3.123712777617911) q[17];
cx q[16],q[17];
ry(2.7848268554775726) q[18];
ry(-0.8319923055143716) q[19];
cx q[18],q[19];
ry(1.7588781861369593) q[18];
ry(-1.4830787644673498) q[19];
cx q[18],q[19];
ry(-1.4959352378775086) q[0];
ry(-0.04247237964472504) q[2];
cx q[0],q[2];
ry(-2.657110671813129) q[0];
ry(2.837675381588769) q[2];
cx q[0],q[2];
ry(1.2551688682092017) q[2];
ry(-1.778752396461794) q[4];
cx q[2],q[4];
ry(3.0458833533132403) q[2];
ry(-3.110104008055734) q[4];
cx q[2],q[4];
ry(-2.175893921619444) q[4];
ry(2.684748283877745) q[6];
cx q[4],q[6];
ry(-3.1406562398764835) q[4];
ry(-3.051281963073608) q[6];
cx q[4],q[6];
ry(1.9070822574708945) q[6];
ry(0.9876659296599959) q[8];
cx q[6],q[8];
ry(3.1158965161486405) q[6];
ry(0.003185823000229071) q[8];
cx q[6],q[8];
ry(-1.5559168333308655) q[8];
ry(-2.710539065661764) q[10];
cx q[8],q[10];
ry(0.07458117751933457) q[8];
ry(3.0752351784413667) q[10];
cx q[8],q[10];
ry(-0.11393219132710819) q[10];
ry(-0.7966103006935157) q[12];
cx q[10],q[12];
ry(3.1404655019271965) q[10];
ry(3.1406132800797235) q[12];
cx q[10],q[12];
ry(2.4837469096320666) q[12];
ry(-2.6010572603081634) q[14];
cx q[12],q[14];
ry(0.1547085355344868) q[12];
ry(-2.1431652664124865) q[14];
cx q[12],q[14];
ry(0.4296123859159824) q[14];
ry(-1.7515082133747004) q[16];
cx q[14],q[16];
ry(-0.055854110527161716) q[14];
ry(-0.585652462793093) q[16];
cx q[14],q[16];
ry(0.18252728315830424) q[16];
ry(-2.6990517112733854) q[18];
cx q[16],q[18];
ry(1.3300190760494512) q[16];
ry(3.1357154925973476) q[18];
cx q[16],q[18];
ry(-2.376142196268143) q[1];
ry(2.3207549622202146) q[3];
cx q[1],q[3];
ry(-0.5533967360872776) q[1];
ry(-3.128449798841573) q[3];
cx q[1],q[3];
ry(1.9433301645705767) q[3];
ry(-2.5950142430579333) q[5];
cx q[3],q[5];
ry(-0.0601956404612344) q[3];
ry(-3.126480799338393) q[5];
cx q[3],q[5];
ry(-0.6615882887333168) q[5];
ry(1.5467543659109637) q[7];
cx q[5],q[7];
ry(-3.125935630914463) q[5];
ry(0.036153483079662596) q[7];
cx q[5],q[7];
ry(0.15288426433623603) q[7];
ry(-2.219393237938893) q[9];
cx q[7],q[9];
ry(-3.1392232815688446) q[7];
ry(3.122848870221586) q[9];
cx q[7],q[9];
ry(-1.6766258553880384) q[9];
ry(-2.2901662267130307) q[11];
cx q[9],q[11];
ry(0.0006123756866811547) q[9];
ry(-3.0399074075106127) q[11];
cx q[9],q[11];
ry(-2.6421693098821386) q[11];
ry(1.025338357963635) q[13];
cx q[11],q[13];
ry(3.1413400515457095) q[11];
ry(-3.1412177739866616) q[13];
cx q[11],q[13];
ry(0.4147464459566432) q[13];
ry(-0.39634460423508244) q[15];
cx q[13],q[15];
ry(-0.13093114934874683) q[13];
ry(-0.9431200953438116) q[15];
cx q[13],q[15];
ry(0.4830849795428831) q[15];
ry(-1.2210381868708335) q[17];
cx q[15],q[17];
ry(-0.0023058895959431397) q[15];
ry(0.055714716448195124) q[17];
cx q[15],q[17];
ry(-1.4216168225085) q[17];
ry(-2.1879861969378522) q[19];
cx q[17],q[19];
ry(-3.0631826721021587) q[17];
ry(0.08258201586799085) q[19];
cx q[17],q[19];
ry(0.8707447266118142) q[0];
ry(1.0271337191278647) q[1];
cx q[0],q[1];
ry(0.8857463960405516) q[0];
ry(3.0857968010725023) q[1];
cx q[0],q[1];
ry(-2.135322758217587) q[2];
ry(0.5614779809381704) q[3];
cx q[2],q[3];
ry(-0.7007606050256657) q[2];
ry(-2.6967819676442755) q[3];
cx q[2],q[3];
ry(0.47045725151212814) q[4];
ry(-1.66243636671111) q[5];
cx q[4],q[5];
ry(0.4780186113150934) q[4];
ry(0.07572068997106653) q[5];
cx q[4],q[5];
ry(-1.5864180430720678) q[6];
ry(0.8260831991502284) q[7];
cx q[6],q[7];
ry(-1.4409245683520489) q[6];
ry(0.3270131763120858) q[7];
cx q[6],q[7];
ry(-1.1621801512039287) q[8];
ry(-1.3735669716745789) q[9];
cx q[8],q[9];
ry(2.3554472539863456) q[8];
ry(-0.3583643626840657) q[9];
cx q[8],q[9];
ry(-2.3696487810951115) q[10];
ry(1.5287214325473548) q[11];
cx q[10],q[11];
ry(-2.5919310001136613) q[10];
ry(-2.483888931271972) q[11];
cx q[10],q[11];
ry(-1.450233901328689) q[12];
ry(1.8993311398360602) q[13];
cx q[12],q[13];
ry(-0.06943774694792913) q[12];
ry(-1.5746668328008389) q[13];
cx q[12],q[13];
ry(1.278702044806912) q[14];
ry(-2.6098272183646976) q[15];
cx q[14],q[15];
ry(-0.0014147225406904817) q[14];
ry(-3.1399904013557838) q[15];
cx q[14],q[15];
ry(-1.19935468454971) q[16];
ry(1.6390419473542674) q[17];
cx q[16],q[17];
ry(1.7031621154257524) q[16];
ry(-3.057512312755586) q[17];
cx q[16],q[17];
ry(1.3594254994498174) q[18];
ry(2.3873484729160843) q[19];
cx q[18],q[19];
ry(-1.775674845094132) q[18];
ry(1.780652680967499) q[19];
cx q[18],q[19];
ry(-0.3401012737491209) q[0];
ry(1.6777795545021583) q[2];
cx q[0],q[2];
ry(-0.048046750023147844) q[0];
ry(3.047924017476871) q[2];
cx q[0],q[2];
ry(-1.7613504510292826) q[2];
ry(1.0136880491802227) q[4];
cx q[2],q[4];
ry(-3.1169411047622604) q[2];
ry(0.02829025208669034) q[4];
cx q[2],q[4];
ry(-1.9231767975749539) q[4];
ry(2.2740616995265404) q[6];
cx q[4],q[6];
ry(-3.1318908503875806) q[4];
ry(-0.06353021108953971) q[6];
cx q[4],q[6];
ry(1.324371174618351) q[6];
ry(1.8220686038632397) q[8];
cx q[6],q[8];
ry(-3.111496649456351) q[6];
ry(0.019554914451042293) q[8];
cx q[6],q[8];
ry(-2.3599493226253467) q[8];
ry(0.9704783759935383) q[10];
cx q[8],q[10];
ry(-0.06981725950480566) q[8];
ry(-0.040899148045211) q[10];
cx q[8],q[10];
ry(-2.789389412794756) q[10];
ry(-2.2194075716876176) q[12];
cx q[10],q[12];
ry(-3.117736238049383) q[10];
ry(-3.1329489862388114) q[12];
cx q[10],q[12];
ry(0.10422800632172624) q[12];
ry(2.5964371483186954) q[14];
cx q[12],q[14];
ry(-3.1377811923148524) q[12];
ry(-3.1359129306551816) q[14];
cx q[12],q[14];
ry(-1.8343137821189686) q[14];
ry(1.575111003542438) q[16];
cx q[14],q[16];
ry(0.001700226863601989) q[14];
ry(-0.5824851346048066) q[16];
cx q[14],q[16];
ry(-0.4335858260872621) q[16];
ry(0.23738394190698564) q[18];
cx q[16],q[18];
ry(-0.0019171244660673992) q[16];
ry(3.1318955954705654) q[18];
cx q[16],q[18];
ry(-0.8164524329705612) q[1];
ry(2.555126380840857) q[3];
cx q[1],q[3];
ry(-3.108333966995271) q[1];
ry(-0.04889785228814445) q[3];
cx q[1],q[3];
ry(-0.006423584897028434) q[3];
ry(0.31071614173418716) q[5];
cx q[3],q[5];
ry(-0.07277690750670704) q[3];
ry(-0.0020528917062767604) q[5];
cx q[3],q[5];
ry(0.4498732855187896) q[5];
ry(1.9001567670953836) q[7];
cx q[5],q[7];
ry(-3.118484912824065) q[5];
ry(-0.018859645793094515) q[7];
cx q[5],q[7];
ry(2.206174809082091) q[7];
ry(1.0896671966537534) q[9];
cx q[7],q[9];
ry(0.0230257376212748) q[7];
ry(-0.029846337846812798) q[9];
cx q[7],q[9];
ry(0.6976841052735329) q[9];
ry(-1.8383025549761527) q[11];
cx q[9],q[11];
ry(3.121845661578879) q[9];
ry(-0.08418952571139204) q[11];
cx q[9],q[11];
ry(-0.30629383897945045) q[11];
ry(0.45577159455617355) q[13];
cx q[11],q[13];
ry(-0.005473614246448131) q[11];
ry(3.1354807341945348) q[13];
cx q[11],q[13];
ry(2.4610223697878517) q[13];
ry(-2.1026655680670543) q[15];
cx q[13],q[15];
ry(-1.543014023955723) q[13];
ry(3.134026595092916) q[15];
cx q[13],q[15];
ry(-1.3384609037188326) q[15];
ry(-2.8879646651151227) q[17];
cx q[15],q[17];
ry(-0.007290822509207486) q[15];
ry(0.01586831046562409) q[17];
cx q[15],q[17];
ry(1.1527396823429434) q[17];
ry(-0.39655324643583706) q[19];
cx q[17],q[19];
ry(3.125806213486689) q[17];
ry(-0.004407365538917851) q[19];
cx q[17],q[19];
ry(-0.17387582270488117) q[0];
ry(1.21405762153662) q[1];
ry(1.2535078622930005) q[2];
ry(-2.687403161909947) q[3];
ry(2.109295120964254) q[4];
ry(2.512086047161798) q[5];
ry(-2.602910801880224) q[6];
ry(-0.7809232380939397) q[7];
ry(1.3605343551462383) q[8];
ry(1.8842252876015273) q[9];
ry(1.9533353909014177) q[10];
ry(2.17835623764592) q[11];
ry(-2.03320576242793) q[12];
ry(2.7505476858546785) q[13];
ry(0.9197865161002391) q[14];
ry(-1.3283192881419872) q[15];
ry(1.7602597131990567) q[16];
ry(2.664224864821144) q[17];
ry(0.6347999924021926) q[18];
ry(-2.5924632935124516) q[19];