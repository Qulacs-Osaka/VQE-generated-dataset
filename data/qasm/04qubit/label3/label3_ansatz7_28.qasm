OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.9524651248576594) q[0];
ry(-1.0915662974556957) q[1];
cx q[0],q[1];
ry(0.11789682387391467) q[0];
ry(-2.560289373794491) q[1];
cx q[0],q[1];
ry(-1.0150582070846736) q[0];
ry(-3.095493988976297) q[2];
cx q[0],q[2];
ry(-1.6121897745511513) q[0];
ry(-1.3617916376330792) q[2];
cx q[0],q[2];
ry(-0.22315351846415918) q[0];
ry(-0.5049795851748495) q[3];
cx q[0],q[3];
ry(0.7572265776606608) q[0];
ry(2.932079149848705) q[3];
cx q[0],q[3];
ry(-0.14379745040528988) q[1];
ry(0.5555060564636737) q[2];
cx q[1],q[2];
ry(-1.2251981957799751) q[1];
ry(1.5557522749489312) q[2];
cx q[1],q[2];
ry(0.9276793724840813) q[1];
ry(0.5220559799899172) q[3];
cx q[1],q[3];
ry(2.4464041685576667) q[1];
ry(2.367233408274831) q[3];
cx q[1],q[3];
ry(-0.6629968129671298) q[2];
ry(-2.7688232454435466) q[3];
cx q[2],q[3];
ry(-1.120669062106104) q[2];
ry(2.219589887138983) q[3];
cx q[2],q[3];
ry(-1.0135725866641452) q[0];
ry(-0.4303809791068567) q[1];
cx q[0],q[1];
ry(-2.005537891296278) q[0];
ry(0.8277998307021582) q[1];
cx q[0],q[1];
ry(0.29691451921534273) q[0];
ry(2.1321356243011387) q[2];
cx q[0],q[2];
ry(-1.148946751869745) q[0];
ry(-2.668649899790988) q[2];
cx q[0],q[2];
ry(-2.6530932352586234) q[0];
ry(-2.8029666354608924) q[3];
cx q[0],q[3];
ry(2.915576699562858) q[0];
ry(2.2000798998622706) q[3];
cx q[0],q[3];
ry(-1.7992707349147574) q[1];
ry(-2.02269129305881) q[2];
cx q[1],q[2];
ry(-1.5218736363335914) q[1];
ry(0.9448460717329814) q[2];
cx q[1],q[2];
ry(2.377003040777039) q[1];
ry(-0.30263195288139855) q[3];
cx q[1],q[3];
ry(2.170581893084413) q[1];
ry(1.9751981059111765) q[3];
cx q[1],q[3];
ry(2.516592605749437) q[2];
ry(2.3496809055435572) q[3];
cx q[2],q[3];
ry(1.1747268119523495) q[2];
ry(-1.8301732698621924) q[3];
cx q[2],q[3];
ry(2.095981943313958) q[0];
ry(0.8618405571858041) q[1];
cx q[0],q[1];
ry(0.8125515746899552) q[0];
ry(1.1506064447647049) q[1];
cx q[0],q[1];
ry(2.127024607105997) q[0];
ry(1.3538334867453372) q[2];
cx q[0],q[2];
ry(1.4310069318955252) q[0];
ry(0.016882397247835502) q[2];
cx q[0],q[2];
ry(0.8324888552279792) q[0];
ry(1.0318683376231255) q[3];
cx q[0],q[3];
ry(-0.9432771451509738) q[0];
ry(-0.6954922615118395) q[3];
cx q[0],q[3];
ry(-1.7609542928025845) q[1];
ry(-1.285606691176496) q[2];
cx q[1],q[2];
ry(1.4908286738114873) q[1];
ry(-0.09039059817710657) q[2];
cx q[1],q[2];
ry(-1.692142526962367) q[1];
ry(-1.0396994330730314) q[3];
cx q[1],q[3];
ry(0.7782384300016735) q[1];
ry(1.418977912188227) q[3];
cx q[1],q[3];
ry(-3.1331602759752895) q[2];
ry(-1.90476141221529) q[3];
cx q[2],q[3];
ry(2.8632266196749074) q[2];
ry(2.926756356854335) q[3];
cx q[2],q[3];
ry(-2.463948059970906) q[0];
ry(-0.2948165828492079) q[1];
cx q[0],q[1];
ry(1.0145245801377305) q[0];
ry(0.30541530924067367) q[1];
cx q[0],q[1];
ry(-1.794871147682632) q[0];
ry(-2.949303335877342) q[2];
cx q[0],q[2];
ry(-1.7353458975150595) q[0];
ry(-0.03584494685925596) q[2];
cx q[0],q[2];
ry(2.6912292032456184) q[0];
ry(0.042408813983683125) q[3];
cx q[0],q[3];
ry(0.07765557837333503) q[0];
ry(2.8889678981621207) q[3];
cx q[0],q[3];
ry(-2.1373312399852438) q[1];
ry(-2.807460581294185) q[2];
cx q[1],q[2];
ry(0.9643055172013915) q[1];
ry(-2.7646692395320414) q[2];
cx q[1],q[2];
ry(-1.2576812817669856) q[1];
ry(0.27444325145332904) q[3];
cx q[1],q[3];
ry(1.6799392022569635) q[1];
ry(2.9192586283172104) q[3];
cx q[1],q[3];
ry(-2.321291584369295) q[2];
ry(1.0379246489924672) q[3];
cx q[2],q[3];
ry(-1.1476217343560817) q[2];
ry(1.6960464999158036) q[3];
cx q[2],q[3];
ry(0.8423135945392062) q[0];
ry(-0.7435890697776308) q[1];
cx q[0],q[1];
ry(0.8288202065631372) q[0];
ry(-0.04600115963853124) q[1];
cx q[0],q[1];
ry(-2.8622648857808204) q[0];
ry(-1.3772423306996444) q[2];
cx q[0],q[2];
ry(2.1125440641090787) q[0];
ry(2.8430623867385942) q[2];
cx q[0],q[2];
ry(-1.142190725277214) q[0];
ry(2.993971756593905) q[3];
cx q[0],q[3];
ry(1.3576552571969887) q[0];
ry(0.9817911375524679) q[3];
cx q[0],q[3];
ry(0.4765043504940233) q[1];
ry(2.134366049803565) q[2];
cx q[1],q[2];
ry(-1.3684205383405483) q[1];
ry(-0.8324049664803745) q[2];
cx q[1],q[2];
ry(-0.7075663854039238) q[1];
ry(-0.682777004834878) q[3];
cx q[1],q[3];
ry(-2.3302435750090864) q[1];
ry(-3.0079188580534812) q[3];
cx q[1],q[3];
ry(0.015597350171221437) q[2];
ry(0.975582996309976) q[3];
cx q[2],q[3];
ry(-2.5396656609664965) q[2];
ry(2.131952486408224) q[3];
cx q[2],q[3];
ry(2.389698018274365) q[0];
ry(-3.1091457941496667) q[1];
cx q[0],q[1];
ry(1.2096317670574521) q[0];
ry(-1.147491887742775) q[1];
cx q[0],q[1];
ry(1.4328451339857544) q[0];
ry(-2.113731677689538) q[2];
cx q[0],q[2];
ry(-1.8869307615679631) q[0];
ry(3.0239865221383013) q[2];
cx q[0],q[2];
ry(1.7103839548058453) q[0];
ry(-2.1462132328318013) q[3];
cx q[0],q[3];
ry(-0.4162414043033103) q[0];
ry(2.500075087815484) q[3];
cx q[0],q[3];
ry(-0.1995782669304221) q[1];
ry(-1.4371987017456314) q[2];
cx q[1],q[2];
ry(0.7761820428177915) q[1];
ry(-0.2791942395446485) q[2];
cx q[1],q[2];
ry(2.7127524050768783) q[1];
ry(2.2510067850476894) q[3];
cx q[1],q[3];
ry(1.8731948368160827) q[1];
ry(-0.3303088673271324) q[3];
cx q[1],q[3];
ry(2.087098588803719) q[2];
ry(0.4087900334947756) q[3];
cx q[2],q[3];
ry(-1.887171096506588) q[2];
ry(-3.0049893867191204) q[3];
cx q[2],q[3];
ry(-1.5847379310486236) q[0];
ry(1.9135265193122422) q[1];
cx q[0],q[1];
ry(2.5581016500767553) q[0];
ry(0.4710059923026125) q[1];
cx q[0],q[1];
ry(-0.32064984185677226) q[0];
ry(-0.6020996874572224) q[2];
cx q[0],q[2];
ry(0.5383124582392673) q[0];
ry(-1.5019306501400669) q[2];
cx q[0],q[2];
ry(1.2088306244147375) q[0];
ry(2.0576976787617762) q[3];
cx q[0],q[3];
ry(0.1823487892342941) q[0];
ry(1.2172386446887358) q[3];
cx q[0],q[3];
ry(-0.6238024399607649) q[1];
ry(1.3380764557201512) q[2];
cx q[1],q[2];
ry(0.09244450346826233) q[1];
ry(-2.7944586538634613) q[2];
cx q[1],q[2];
ry(-2.819272473665649) q[1];
ry(1.0268568857365672) q[3];
cx q[1],q[3];
ry(-2.1979137561888797) q[1];
ry(-0.3920326631851448) q[3];
cx q[1],q[3];
ry(0.35095063778618574) q[2];
ry(-1.9613598251134334) q[3];
cx q[2],q[3];
ry(2.8802099324849) q[2];
ry(1.1604988874379971) q[3];
cx q[2],q[3];
ry(1.6532982880998843) q[0];
ry(2.5659431805152186) q[1];
cx q[0],q[1];
ry(1.5057882618731986) q[0];
ry(0.9042619367164774) q[1];
cx q[0],q[1];
ry(-0.7496513150795819) q[0];
ry(0.24531107439969022) q[2];
cx q[0],q[2];
ry(2.4039748289906577) q[0];
ry(-0.26305578451274947) q[2];
cx q[0],q[2];
ry(2.674444340191671) q[0];
ry(2.2127771984078732) q[3];
cx q[0],q[3];
ry(0.8400564213180077) q[0];
ry(-2.495271369226171) q[3];
cx q[0],q[3];
ry(-3.0699183527674774) q[1];
ry(1.9075017241607446) q[2];
cx q[1],q[2];
ry(2.7498894951190707) q[1];
ry(0.71266750099345) q[2];
cx q[1],q[2];
ry(2.8675258532874963) q[1];
ry(-1.2891446032877854) q[3];
cx q[1],q[3];
ry(1.5056881557854194) q[1];
ry(0.7651458716782171) q[3];
cx q[1],q[3];
ry(0.8444405753808231) q[2];
ry(2.1198991231042914) q[3];
cx q[2],q[3];
ry(2.3056020600228404) q[2];
ry(0.05518117607470607) q[3];
cx q[2],q[3];
ry(-3.0899240380046202) q[0];
ry(-0.2078641201461695) q[1];
cx q[0],q[1];
ry(1.8731802025794693) q[0];
ry(-0.033917288414973434) q[1];
cx q[0],q[1];
ry(-0.5899682301585237) q[0];
ry(-2.6452406515796603) q[2];
cx q[0],q[2];
ry(0.7012848256190739) q[0];
ry(1.5805583078736745) q[2];
cx q[0],q[2];
ry(-1.6360207894307388) q[0];
ry(-0.18908108842282698) q[3];
cx q[0],q[3];
ry(-0.9193247295018093) q[0];
ry(-1.0382092749645846) q[3];
cx q[0],q[3];
ry(0.7893013766258636) q[1];
ry(-3.1213119429953893) q[2];
cx q[1],q[2];
ry(-0.4102888765301296) q[1];
ry(-0.5881922134709834) q[2];
cx q[1],q[2];
ry(1.8681758354729097) q[1];
ry(-0.9887411409748115) q[3];
cx q[1],q[3];
ry(1.2156664811894944) q[1];
ry(2.2341087730556386) q[3];
cx q[1],q[3];
ry(0.1042053239771279) q[2];
ry(0.12840787582854093) q[3];
cx q[2],q[3];
ry(3.000397919796154) q[2];
ry(2.2746630129193646) q[3];
cx q[2],q[3];
ry(1.9900427844177973) q[0];
ry(1.13951324655903) q[1];
cx q[0],q[1];
ry(-2.877184766368968) q[0];
ry(-2.58366624792833) q[1];
cx q[0],q[1];
ry(-0.7972358294109716) q[0];
ry(0.8150651378548192) q[2];
cx q[0],q[2];
ry(-0.7494191008290835) q[0];
ry(-0.9509469399666903) q[2];
cx q[0],q[2];
ry(-1.1358053063857745) q[0];
ry(0.32028346760727994) q[3];
cx q[0],q[3];
ry(0.249099482659553) q[0];
ry(1.1062619540964342) q[3];
cx q[0],q[3];
ry(1.1089879923693222) q[1];
ry(1.9781475789217637) q[2];
cx q[1],q[2];
ry(-0.10636110589735717) q[1];
ry(-0.7649939090791253) q[2];
cx q[1],q[2];
ry(1.1404852580163687) q[1];
ry(2.214714964384854) q[3];
cx q[1],q[3];
ry(0.9495798564051341) q[1];
ry(-2.6422491831219235) q[3];
cx q[1],q[3];
ry(-2.9180564275195) q[2];
ry(-1.1912062396553682) q[3];
cx q[2],q[3];
ry(2.9448919726980582) q[2];
ry(0.05942528678283633) q[3];
cx q[2],q[3];
ry(1.040114617354351) q[0];
ry(-1.9591927012000443) q[1];
cx q[0],q[1];
ry(3.0402545077895096) q[0];
ry(0.8668252646650483) q[1];
cx q[0],q[1];
ry(1.1639162178761655) q[0];
ry(-1.291374136762876) q[2];
cx q[0],q[2];
ry(1.9976221722518548) q[0];
ry(0.2447629173450112) q[2];
cx q[0],q[2];
ry(-1.8252440909856265) q[0];
ry(-0.5622547541989931) q[3];
cx q[0],q[3];
ry(-0.7963056847147635) q[0];
ry(-2.995349954983255) q[3];
cx q[0],q[3];
ry(-2.3065659096098616) q[1];
ry(-0.06978685837400907) q[2];
cx q[1],q[2];
ry(-1.512498238486994) q[1];
ry(-0.17704476705092811) q[2];
cx q[1],q[2];
ry(-2.3135923572858434) q[1];
ry(2.9115456270034743) q[3];
cx q[1],q[3];
ry(1.6292950612728783) q[1];
ry(-1.2223998168086394) q[3];
cx q[1],q[3];
ry(2.2688954930856804) q[2];
ry(2.7654211548077603) q[3];
cx q[2],q[3];
ry(-2.7581034328620126) q[2];
ry(0.8916869072945662) q[3];
cx q[2],q[3];
ry(-0.7460105302254829) q[0];
ry(1.876441334845667) q[1];
cx q[0],q[1];
ry(-2.4601437023144386) q[0];
ry(-2.945133697971734) q[1];
cx q[0],q[1];
ry(-2.8447612003201503) q[0];
ry(1.7967613282602064) q[2];
cx q[0],q[2];
ry(1.1201669801397434) q[0];
ry(3.0600307137373006) q[2];
cx q[0],q[2];
ry(1.9038456635420564) q[0];
ry(-2.1074607012579234) q[3];
cx q[0],q[3];
ry(-0.26827576828461686) q[0];
ry(-1.2327471766379592) q[3];
cx q[0],q[3];
ry(2.843620805769613) q[1];
ry(-1.4430314451523492) q[2];
cx q[1],q[2];
ry(2.862432359890093) q[1];
ry(-0.9145356543711194) q[2];
cx q[1],q[2];
ry(-0.46885340309785756) q[1];
ry(-3.0271674800402235) q[3];
cx q[1],q[3];
ry(-1.679262574900342) q[1];
ry(2.6028627835441225) q[3];
cx q[1],q[3];
ry(-1.1800937034869925) q[2];
ry(-2.6861671001046634) q[3];
cx q[2],q[3];
ry(0.6675503091983036) q[2];
ry(3.02448226323816) q[3];
cx q[2],q[3];
ry(-1.0439740601358154) q[0];
ry(1.1961435320231866) q[1];
cx q[0],q[1];
ry(1.1154630985711178) q[0];
ry(3.00364506852998) q[1];
cx q[0],q[1];
ry(-0.572283172296807) q[0];
ry(-1.0291286796090164) q[2];
cx q[0],q[2];
ry(-2.407575586603256) q[0];
ry(2.178410346459281) q[2];
cx q[0],q[2];
ry(-1.5227282983709634) q[0];
ry(3.1394710710864975) q[3];
cx q[0],q[3];
ry(-1.739168967919456) q[0];
ry(-1.0514593615329302) q[3];
cx q[0],q[3];
ry(-1.6760854115995187) q[1];
ry(-1.2723003860926196) q[2];
cx q[1],q[2];
ry(2.0456622109717157) q[1];
ry(-2.8881420152072144) q[2];
cx q[1],q[2];
ry(-3.0032943481731285) q[1];
ry(2.1994091023965003) q[3];
cx q[1],q[3];
ry(-1.2441041426237585) q[1];
ry(1.4620411711635841) q[3];
cx q[1],q[3];
ry(-2.4554335973796406) q[2];
ry(-1.9238414236409158) q[3];
cx q[2],q[3];
ry(1.596404591748603) q[2];
ry(1.5184213481634599) q[3];
cx q[2],q[3];
ry(-1.6950753712631121) q[0];
ry(1.3899625846346049) q[1];
cx q[0],q[1];
ry(-1.4021987268493543) q[0];
ry(1.67414871037692) q[1];
cx q[0],q[1];
ry(0.18265057871442655) q[0];
ry(2.8755655902182946) q[2];
cx q[0],q[2];
ry(-1.1576663906345954) q[0];
ry(0.5619052004493149) q[2];
cx q[0],q[2];
ry(-3.061459325147397) q[0];
ry(1.0892182409063231) q[3];
cx q[0],q[3];
ry(0.6176307063119514) q[0];
ry(-2.080158925181243) q[3];
cx q[0],q[3];
ry(-1.6953084499965576) q[1];
ry(-2.887779645636814) q[2];
cx q[1],q[2];
ry(-2.9478895189030236) q[1];
ry(-1.1699311111182655) q[2];
cx q[1],q[2];
ry(0.7971975863270542) q[1];
ry(1.2249291640450144) q[3];
cx q[1],q[3];
ry(-2.563783588268909) q[1];
ry(2.104673703235034) q[3];
cx q[1],q[3];
ry(0.3389296930028305) q[2];
ry(1.3937703553315108) q[3];
cx q[2],q[3];
ry(-1.8077923830727025) q[2];
ry(-1.2708115453367475) q[3];
cx q[2],q[3];
ry(-1.8776233700848817) q[0];
ry(-2.2372493373822397) q[1];
cx q[0],q[1];
ry(0.4860093457423145) q[0];
ry(-1.4172081626294224) q[1];
cx q[0],q[1];
ry(0.017479437913473284) q[0];
ry(1.1495310549390192) q[2];
cx q[0],q[2];
ry(-1.3031188419721318) q[0];
ry(-0.8214681884669002) q[2];
cx q[0],q[2];
ry(-2.3576236889317523) q[0];
ry(0.19868456369903517) q[3];
cx q[0],q[3];
ry(-1.880838006086639) q[0];
ry(-1.969843369651709) q[3];
cx q[0],q[3];
ry(0.1270298382555426) q[1];
ry(0.19655995175273588) q[2];
cx q[1],q[2];
ry(0.6729735974848203) q[1];
ry(-0.9819877520289744) q[2];
cx q[1],q[2];
ry(-0.5174044843217899) q[1];
ry(-2.293127202369835) q[3];
cx q[1],q[3];
ry(-1.2357997816357926) q[1];
ry(-2.6672470769944905) q[3];
cx q[1],q[3];
ry(-3.123581473427262) q[2];
ry(0.9529078045016988) q[3];
cx q[2],q[3];
ry(-2.4448117479309563) q[2];
ry(-0.5260084895880688) q[3];
cx q[2],q[3];
ry(-2.947942054715097) q[0];
ry(-1.8382424666363033) q[1];
cx q[0],q[1];
ry(1.6842114442536784) q[0];
ry(1.3901866075617324) q[1];
cx q[0],q[1];
ry(-0.7392045810252037) q[0];
ry(1.2603846329577575) q[2];
cx q[0],q[2];
ry(2.9821465574469994) q[0];
ry(-0.49588241195534366) q[2];
cx q[0],q[2];
ry(1.9000164769234091) q[0];
ry(1.577186847022795) q[3];
cx q[0],q[3];
ry(-2.350118817771191) q[0];
ry(2.619011916351176) q[3];
cx q[0],q[3];
ry(2.3584476434450417) q[1];
ry(-1.317297893558326) q[2];
cx q[1],q[2];
ry(0.45471714093301663) q[1];
ry(0.5006657962465866) q[2];
cx q[1],q[2];
ry(-2.057378441602106) q[1];
ry(-1.8807159469126737) q[3];
cx q[1],q[3];
ry(-1.6149875218207517) q[1];
ry(-1.3851024710470776) q[3];
cx q[1],q[3];
ry(-0.6604184328581201) q[2];
ry(1.491634288067418) q[3];
cx q[2],q[3];
ry(1.9431908856234767) q[2];
ry(-2.019783049249571) q[3];
cx q[2],q[3];
ry(1.2878489475988246) q[0];
ry(-1.351569538219273) q[1];
cx q[0],q[1];
ry(1.23637225807797) q[0];
ry(3.0020383427911446) q[1];
cx q[0],q[1];
ry(-3.012588309449255) q[0];
ry(0.036850579228503655) q[2];
cx q[0],q[2];
ry(3.0127018341947065) q[0];
ry(0.4140901355847979) q[2];
cx q[0],q[2];
ry(-1.1348797519689098) q[0];
ry(0.7403810212835422) q[3];
cx q[0],q[3];
ry(1.2900540650143713) q[0];
ry(-0.4767678328634016) q[3];
cx q[0],q[3];
ry(0.5506722892857391) q[1];
ry(-2.9494341794025587) q[2];
cx q[1],q[2];
ry(-0.3275155701092194) q[1];
ry(1.7437228618663543) q[2];
cx q[1],q[2];
ry(-2.184228283785195) q[1];
ry(0.8226658199823742) q[3];
cx q[1],q[3];
ry(1.712251972290363) q[1];
ry(3.135907206163085) q[3];
cx q[1],q[3];
ry(-0.49379674009958274) q[2];
ry(0.3388383618015896) q[3];
cx q[2],q[3];
ry(3.026798921806474) q[2];
ry(-2.0473326355783916) q[3];
cx q[2],q[3];
ry(-2.4528030839089072) q[0];
ry(-0.12422287647642172) q[1];
cx q[0],q[1];
ry(-0.9552892762741516) q[0];
ry(-1.070270332356797) q[1];
cx q[0],q[1];
ry(0.33496098906798755) q[0];
ry(-2.667720082716813) q[2];
cx q[0],q[2];
ry(-2.248984162341915) q[0];
ry(2.756505123303593) q[2];
cx q[0],q[2];
ry(-0.7214754375964058) q[0];
ry(-2.5852951519879346) q[3];
cx q[0],q[3];
ry(1.6360249664184057) q[0];
ry(2.68304297319911) q[3];
cx q[0],q[3];
ry(-0.40313690892421405) q[1];
ry(0.4761561739188256) q[2];
cx q[1],q[2];
ry(1.6503598056359197) q[1];
ry(1.2165259380188678) q[2];
cx q[1],q[2];
ry(-2.8812747894086836) q[1];
ry(-0.5269627581489553) q[3];
cx q[1],q[3];
ry(2.543190924526595) q[1];
ry(-0.9783701035239198) q[3];
cx q[1],q[3];
ry(1.6558461514562044) q[2];
ry(2.446436982379834) q[3];
cx q[2],q[3];
ry(2.68364053339801) q[2];
ry(1.4373072776694897) q[3];
cx q[2],q[3];
ry(-1.882081200648446) q[0];
ry(-1.1453859223505785) q[1];
cx q[0],q[1];
ry(0.9880594719032484) q[0];
ry(0.9825012070748613) q[1];
cx q[0],q[1];
ry(2.759895254416635) q[0];
ry(2.5798378485353393) q[2];
cx q[0],q[2];
ry(2.334759141002322) q[0];
ry(0.6786081005799813) q[2];
cx q[0],q[2];
ry(-1.4703704631592762) q[0];
ry(-2.0379219606485846) q[3];
cx q[0],q[3];
ry(1.7382713728583974) q[0];
ry(1.4321317028126883) q[3];
cx q[0],q[3];
ry(-2.505945445521093) q[1];
ry(-2.2073526363828515) q[2];
cx q[1],q[2];
ry(1.8604972425206645) q[1];
ry(2.5693785724327345) q[2];
cx q[1],q[2];
ry(-1.2005167374570185) q[1];
ry(-3.0158765195633292) q[3];
cx q[1],q[3];
ry(-1.6396007519818943) q[1];
ry(-0.63906319022602) q[3];
cx q[1],q[3];
ry(2.830365632937276) q[2];
ry(1.4961219280991038) q[3];
cx q[2],q[3];
ry(-1.1567116121636403) q[2];
ry(-0.7424951489713667) q[3];
cx q[2],q[3];
ry(0.8077766052714069) q[0];
ry(1.8221398928143566) q[1];
cx q[0],q[1];
ry(2.881448674714058) q[0];
ry(-1.2748799569453309) q[1];
cx q[0],q[1];
ry(-1.6833301756483419) q[0];
ry(-1.381618320505475) q[2];
cx q[0],q[2];
ry(0.20418999752124023) q[0];
ry(-2.0255554619633687) q[2];
cx q[0],q[2];
ry(0.6694987158280847) q[0];
ry(-3.100553250486787) q[3];
cx q[0],q[3];
ry(-3.0233710261086753) q[0];
ry(-2.7265000428342754) q[3];
cx q[0],q[3];
ry(0.3160707149117391) q[1];
ry(-2.179058357751407) q[2];
cx q[1],q[2];
ry(-2.0506035021154068) q[1];
ry(0.5370685600916586) q[2];
cx q[1],q[2];
ry(1.7725377911864177) q[1];
ry(-2.881406618871405) q[3];
cx q[1],q[3];
ry(2.1128982473331153) q[1];
ry(2.811839963169298) q[3];
cx q[1],q[3];
ry(0.9054965902172301) q[2];
ry(-1.870381146218476) q[3];
cx q[2],q[3];
ry(-0.07064617977765586) q[2];
ry(0.6514235320466808) q[3];
cx q[2],q[3];
ry(-2.050939975008462) q[0];
ry(-0.4745322637106956) q[1];
cx q[0],q[1];
ry(-0.3223686376592256) q[0];
ry(-1.580997879171365) q[1];
cx q[0],q[1];
ry(-2.4049842898617406) q[0];
ry(-0.18240758420822975) q[2];
cx q[0],q[2];
ry(-0.35311329556756543) q[0];
ry(1.7449250902580142) q[2];
cx q[0],q[2];
ry(2.831266051244851) q[0];
ry(-0.02377664449854145) q[3];
cx q[0],q[3];
ry(-0.2875443357117458) q[0];
ry(2.2368065139758198) q[3];
cx q[0],q[3];
ry(-2.770718628808992) q[1];
ry(-0.5399435321188057) q[2];
cx q[1],q[2];
ry(-0.48344082405901617) q[1];
ry(3.13759093850067) q[2];
cx q[1],q[2];
ry(2.830836605055836) q[1];
ry(2.3294308038678198) q[3];
cx q[1],q[3];
ry(0.6723378221147751) q[1];
ry(2.743021876560409) q[3];
cx q[1],q[3];
ry(-2.718058644385344) q[2];
ry(-0.85882974156272) q[3];
cx q[2],q[3];
ry(0.22784959651085318) q[2];
ry(-2.0209982541731737) q[3];
cx q[2],q[3];
ry(-0.8940328521522627) q[0];
ry(0.4784938367883731) q[1];
cx q[0],q[1];
ry(0.5753085428011384) q[0];
ry(2.838945156585936) q[1];
cx q[0],q[1];
ry(-1.765876232726888) q[0];
ry(1.0306222226031991) q[2];
cx q[0],q[2];
ry(-0.019116713814805272) q[0];
ry(2.042703328864021) q[2];
cx q[0],q[2];
ry(0.25956312069718646) q[0];
ry(2.531704936760157) q[3];
cx q[0],q[3];
ry(0.0444686899020391) q[0];
ry(-2.322982431653034) q[3];
cx q[0],q[3];
ry(-2.5143408533679565) q[1];
ry(1.369325391686654) q[2];
cx q[1],q[2];
ry(-1.067054598033942) q[1];
ry(-1.7741496018414764) q[2];
cx q[1],q[2];
ry(-2.679310591013904) q[1];
ry(-2.4740272380446626) q[3];
cx q[1],q[3];
ry(2.0631175466801857) q[1];
ry(-2.60392692597263) q[3];
cx q[1],q[3];
ry(0.4664775614649148) q[2];
ry(0.3062010107270563) q[3];
cx q[2],q[3];
ry(-1.995337651658211) q[2];
ry(-1.8053799878380814) q[3];
cx q[2],q[3];
ry(-1.4170279803744457) q[0];
ry(0.6695129580728993) q[1];
cx q[0],q[1];
ry(-0.08011356302069877) q[0];
ry(-0.8173103428445219) q[1];
cx q[0],q[1];
ry(1.0266228695493187) q[0];
ry(-1.902231000812593) q[2];
cx q[0],q[2];
ry(1.0111007137149897) q[0];
ry(1.723294019214091) q[2];
cx q[0],q[2];
ry(-1.9698192464496866) q[0];
ry(-2.1675701725609984) q[3];
cx q[0],q[3];
ry(-0.7900189668229847) q[0];
ry(1.7131580730707479) q[3];
cx q[0],q[3];
ry(0.6170175334229706) q[1];
ry(2.8286450754255017) q[2];
cx q[1],q[2];
ry(1.8140231388614918) q[1];
ry(-2.3805649232842363) q[2];
cx q[1],q[2];
ry(0.03203765669772807) q[1];
ry(-0.02632615904788871) q[3];
cx q[1],q[3];
ry(-3.127268766881021) q[1];
ry(1.2460723229683115) q[3];
cx q[1],q[3];
ry(3.0373603712922606) q[2];
ry(2.9643512003140313) q[3];
cx q[2],q[3];
ry(-1.23527605844731) q[2];
ry(1.5954528591946158) q[3];
cx q[2],q[3];
ry(1.2709462835459764) q[0];
ry(-2.3539140180852143) q[1];
cx q[0],q[1];
ry(-0.05478959994569435) q[0];
ry(-0.8410282279107264) q[1];
cx q[0],q[1];
ry(0.520446151747492) q[0];
ry(-1.7466190608858172) q[2];
cx q[0],q[2];
ry(-2.940915743790028) q[0];
ry(2.541503750715142) q[2];
cx q[0],q[2];
ry(-1.8948329307069378) q[0];
ry(0.011474677953877688) q[3];
cx q[0],q[3];
ry(1.735772871539858) q[0];
ry(-1.4601076429569957) q[3];
cx q[0],q[3];
ry(1.9268761192734125) q[1];
ry(2.8316148074153737) q[2];
cx q[1],q[2];
ry(2.594830802077338) q[1];
ry(-0.6866767643731827) q[2];
cx q[1],q[2];
ry(0.11508091017319183) q[1];
ry(-1.7524321893391979) q[3];
cx q[1],q[3];
ry(2.023136942259164) q[1];
ry(-0.11951429253923886) q[3];
cx q[1],q[3];
ry(-2.1341010203657262) q[2];
ry(0.1520416701024807) q[3];
cx q[2],q[3];
ry(1.6969271956965701) q[2];
ry(0.10859101468780098) q[3];
cx q[2],q[3];
ry(-0.643385237058705) q[0];
ry(0.8975701831830563) q[1];
cx q[0],q[1];
ry(2.555787945149488) q[0];
ry(-1.7113652956402368) q[1];
cx q[0],q[1];
ry(1.0872086650016382) q[0];
ry(2.323619102761096) q[2];
cx q[0],q[2];
ry(-0.24113546897200394) q[0];
ry(0.5210674370115331) q[2];
cx q[0],q[2];
ry(-0.5306617091738186) q[0];
ry(-0.13820694166239011) q[3];
cx q[0],q[3];
ry(1.243393410842474) q[0];
ry(-1.1722275814601844) q[3];
cx q[0],q[3];
ry(1.6122084859617134) q[1];
ry(-0.8930724179952252) q[2];
cx q[1],q[2];
ry(2.404424062055602) q[1];
ry(-0.9685484465593751) q[2];
cx q[1],q[2];
ry(-1.4948968114072292) q[1];
ry(0.284641393623321) q[3];
cx q[1],q[3];
ry(2.488272536306667) q[1];
ry(0.31623808625420397) q[3];
cx q[1],q[3];
ry(2.839939851003905) q[2];
ry(-3.114955863701811) q[3];
cx q[2],q[3];
ry(-2.3479109926486488) q[2];
ry(-0.21307416876671592) q[3];
cx q[2],q[3];
ry(2.8819447663707676) q[0];
ry(-2.725702498935417) q[1];
cx q[0],q[1];
ry(-0.5059207392999623) q[0];
ry(1.3610925035110852) q[1];
cx q[0],q[1];
ry(1.6714164725502956) q[0];
ry(-0.6139907536068973) q[2];
cx q[0],q[2];
ry(1.4950290673792586) q[0];
ry(-2.20131136151649) q[2];
cx q[0],q[2];
ry(-2.749334178808288) q[0];
ry(-0.05429389936064491) q[3];
cx q[0],q[3];
ry(-1.363382120608852) q[0];
ry(-3.1201867367619265) q[3];
cx q[0],q[3];
ry(-0.546282004791709) q[1];
ry(1.6933772259020698) q[2];
cx q[1],q[2];
ry(0.36520065980687166) q[1];
ry(-0.37080002571611304) q[2];
cx q[1],q[2];
ry(-2.793859071068597) q[1];
ry(1.2928624015198737) q[3];
cx q[1],q[3];
ry(0.26933171933415606) q[1];
ry(0.09291193466154282) q[3];
cx q[1],q[3];
ry(-1.1721518260859605) q[2];
ry(2.048406337227256) q[3];
cx q[2],q[3];
ry(2.556952894736528) q[2];
ry(-1.2717508968429112) q[3];
cx q[2],q[3];
ry(0.6216903328517626) q[0];
ry(0.08382923402349145) q[1];
cx q[0],q[1];
ry(1.2716166422922879) q[0];
ry(-2.6553049646355222) q[1];
cx q[0],q[1];
ry(-1.8919837722270083) q[0];
ry(-2.9756321027001325) q[2];
cx q[0],q[2];
ry(-2.938127269013034) q[0];
ry(-2.3073537525321957) q[2];
cx q[0],q[2];
ry(-0.6562415605114068) q[0];
ry(-2.9997318677488285) q[3];
cx q[0],q[3];
ry(-1.7961931083732967) q[0];
ry(1.6797605635413864) q[3];
cx q[0],q[3];
ry(1.2132179924479325) q[1];
ry(1.8849323914021578) q[2];
cx q[1],q[2];
ry(1.615606334410363) q[1];
ry(-2.57132440090692) q[2];
cx q[1],q[2];
ry(0.401428551677352) q[1];
ry(-2.852970776256292) q[3];
cx q[1],q[3];
ry(2.660282935270288) q[1];
ry(1.3364219525576433) q[3];
cx q[1],q[3];
ry(0.7612405351138376) q[2];
ry(0.39007358184677915) q[3];
cx q[2],q[3];
ry(1.5108650763538818) q[2];
ry(2.2304553953804094) q[3];
cx q[2],q[3];
ry(-1.023300305175394) q[0];
ry(-2.188670211400379) q[1];
cx q[0],q[1];
ry(0.35493386580418873) q[0];
ry(-2.5535931982835827) q[1];
cx q[0],q[1];
ry(-1.6649763144547762) q[0];
ry(-2.8880778872794823) q[2];
cx q[0],q[2];
ry(-0.5295629219798448) q[0];
ry(-2.084783849186498) q[2];
cx q[0],q[2];
ry(-1.0128154779671208) q[0];
ry(0.7051586764877005) q[3];
cx q[0],q[3];
ry(-1.4470813876346063) q[0];
ry(0.9182197141391071) q[3];
cx q[0],q[3];
ry(1.1739482595735597) q[1];
ry(-3.058736921392476) q[2];
cx q[1],q[2];
ry(-2.358214325206354) q[1];
ry(-2.5927032121662803) q[2];
cx q[1],q[2];
ry(-1.1450525852356623) q[1];
ry(0.990940795347786) q[3];
cx q[1],q[3];
ry(-2.9484018175516167) q[1];
ry(-2.8824690140072486) q[3];
cx q[1],q[3];
ry(3.0599014404553566) q[2];
ry(-2.729947443229877) q[3];
cx q[2],q[3];
ry(-1.6487473236084402) q[2];
ry(-2.8850269998241704) q[3];
cx q[2],q[3];
ry(2.722265544639459) q[0];
ry(0.6582221426805565) q[1];
cx q[0],q[1];
ry(1.1952268799105914) q[0];
ry(-0.9270357059274444) q[1];
cx q[0],q[1];
ry(2.8439663018058425) q[0];
ry(0.03744215125914252) q[2];
cx q[0],q[2];
ry(0.28885568616671214) q[0];
ry(2.622486124931157) q[2];
cx q[0],q[2];
ry(0.5259561531175789) q[0];
ry(-1.1288257140686406) q[3];
cx q[0],q[3];
ry(0.05089764937387731) q[0];
ry(-0.014853835048185358) q[3];
cx q[0],q[3];
ry(1.51096946254683) q[1];
ry(-0.8550999535872004) q[2];
cx q[1],q[2];
ry(-1.3305914474920886) q[1];
ry(0.6671400331759418) q[2];
cx q[1],q[2];
ry(-0.5719376555590011) q[1];
ry(-2.489456327451754) q[3];
cx q[1],q[3];
ry(-0.5403450107900607) q[1];
ry(0.9840978186457843) q[3];
cx q[1],q[3];
ry(2.976158584800072) q[2];
ry(-2.250738498326511) q[3];
cx q[2],q[3];
ry(2.583372473464966) q[2];
ry(-0.8674575574340038) q[3];
cx q[2],q[3];
ry(-0.47730473317453537) q[0];
ry(-1.8070720914575684) q[1];
cx q[0],q[1];
ry(1.2514227447813606) q[0];
ry(1.9326525277278486) q[1];
cx q[0],q[1];
ry(1.951865279172777) q[0];
ry(-0.37581306060439523) q[2];
cx q[0],q[2];
ry(1.5228333915621717) q[0];
ry(-1.4679604956995196) q[2];
cx q[0],q[2];
ry(1.5936876485874265) q[0];
ry(-1.012520202065411) q[3];
cx q[0],q[3];
ry(0.007366368567065962) q[0];
ry(-3.1351956233120584) q[3];
cx q[0],q[3];
ry(-3.035007682949255) q[1];
ry(1.714583166946986) q[2];
cx q[1],q[2];
ry(0.1526572371690655) q[1];
ry(0.1562468315799892) q[2];
cx q[1],q[2];
ry(-2.8883236771568894) q[1];
ry(-0.33554882169028843) q[3];
cx q[1],q[3];
ry(-3.0425509685394108) q[1];
ry(-0.6926105707487462) q[3];
cx q[1],q[3];
ry(1.111092201690425) q[2];
ry(0.3207073420929625) q[3];
cx q[2],q[3];
ry(-1.4884290320866662) q[2];
ry(2.535438138723912) q[3];
cx q[2],q[3];
ry(-1.5074767642793012) q[0];
ry(-0.3191410831534185) q[1];
cx q[0],q[1];
ry(-1.1257065362716665) q[0];
ry(0.06809450787024662) q[1];
cx q[0],q[1];
ry(2.9634345530493182) q[0];
ry(0.9017649072334324) q[2];
cx q[0],q[2];
ry(-1.0838278782465247) q[0];
ry(-0.3797741379417099) q[2];
cx q[0],q[2];
ry(-0.8847003383283578) q[0];
ry(-0.9491299050694657) q[3];
cx q[0],q[3];
ry(-2.6168256072590688) q[0];
ry(2.0856112075116746) q[3];
cx q[0],q[3];
ry(-2.3759337718660123) q[1];
ry(1.9943369904415704) q[2];
cx q[1],q[2];
ry(2.377344154977103) q[1];
ry(3.0740888874370342) q[2];
cx q[1],q[2];
ry(0.8099598806469714) q[1];
ry(-0.27343018688308035) q[3];
cx q[1],q[3];
ry(-0.9818701375656289) q[1];
ry(2.242937986076221) q[3];
cx q[1],q[3];
ry(-2.0604435106536) q[2];
ry(-1.0941159229248856) q[3];
cx q[2],q[3];
ry(-1.6364204050244164) q[2];
ry(-0.10378725152147082) q[3];
cx q[2],q[3];
ry(-2.1590400689687) q[0];
ry(-0.8161159671557092) q[1];
ry(2.0629293709673444) q[2];
ry(-2.6923361594336503) q[3];