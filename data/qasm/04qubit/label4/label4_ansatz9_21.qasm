OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.6945545897261078) q[0];
ry(-2.4648165489886344) q[1];
cx q[0],q[1];
ry(1.984475141834955) q[0];
ry(3.0081413558936565) q[1];
cx q[0],q[1];
ry(2.719215570439059) q[2];
ry(-0.1407696144437659) q[3];
cx q[2],q[3];
ry(-2.1020088942971418) q[2];
ry(2.6888477505366994) q[3];
cx q[2],q[3];
ry(-2.636634726578524) q[0];
ry(1.9311090682793848) q[2];
cx q[0],q[2];
ry(0.22908279785073749) q[0];
ry(-3.1362006820943167) q[2];
cx q[0],q[2];
ry(1.2616235744744027) q[1];
ry(-1.2080016869683248) q[3];
cx q[1],q[3];
ry(-1.72491029960523) q[1];
ry(1.1866954238871372) q[3];
cx q[1],q[3];
ry(2.0447486312576384) q[0];
ry(-1.4068532642307074) q[3];
cx q[0],q[3];
ry(-1.2913328937458504) q[0];
ry(2.693599503057684) q[3];
cx q[0],q[3];
ry(2.409790238820979) q[1];
ry(-0.935076524674842) q[2];
cx q[1],q[2];
ry(-2.5710662183676223) q[1];
ry(-2.44143205932128) q[2];
cx q[1],q[2];
ry(-2.0175235940705134) q[0];
ry(2.0835143181512095) q[1];
cx q[0],q[1];
ry(-0.1702534434769234) q[0];
ry(2.2140191748443283) q[1];
cx q[0],q[1];
ry(-2.3727417263190627) q[2];
ry(-2.9526084153990135) q[3];
cx q[2],q[3];
ry(-0.768733084738404) q[2];
ry(0.5042189486101645) q[3];
cx q[2],q[3];
ry(-2.3406664618862547) q[0];
ry(2.2856740375245983) q[2];
cx q[0],q[2];
ry(1.4053034332408325) q[0];
ry(1.8374658034352063) q[2];
cx q[0],q[2];
ry(1.6293995487264465) q[1];
ry(2.406407897744771) q[3];
cx q[1],q[3];
ry(2.104520063942064) q[1];
ry(-1.2539799258445798) q[3];
cx q[1],q[3];
ry(2.1624849925777405) q[0];
ry(1.5708662107784006) q[3];
cx q[0],q[3];
ry(-1.005346809707016) q[0];
ry(1.9356668588562334) q[3];
cx q[0],q[3];
ry(1.723381211121677) q[1];
ry(1.786086508963054) q[2];
cx q[1],q[2];
ry(-0.8581152743367744) q[1];
ry(2.3224788650821826) q[2];
cx q[1],q[2];
ry(-0.8184132163209739) q[0];
ry(-1.4785521951982254) q[1];
cx q[0],q[1];
ry(2.968203377102557) q[0];
ry(2.1231602325260313) q[1];
cx q[0],q[1];
ry(2.457083852109871) q[2];
ry(-2.8543990022491417) q[3];
cx q[2],q[3];
ry(-0.04574734100572453) q[2];
ry(-1.3127219297327788) q[3];
cx q[2],q[3];
ry(0.3670268048733547) q[0];
ry(1.5252193325104746) q[2];
cx q[0],q[2];
ry(-0.020474543974128772) q[0];
ry(-2.998753430003681) q[2];
cx q[0],q[2];
ry(-1.4004293701207267) q[1];
ry(0.00781714908280673) q[3];
cx q[1],q[3];
ry(-2.1852703107223066) q[1];
ry(0.41510770426358906) q[3];
cx q[1],q[3];
ry(-2.6131860288478093) q[0];
ry(-2.4426560177174843) q[3];
cx q[0],q[3];
ry(-1.781911157474846) q[0];
ry(-1.9770947297676436) q[3];
cx q[0],q[3];
ry(0.7073450234968686) q[1];
ry(2.731998381823367) q[2];
cx q[1],q[2];
ry(0.7216328408664527) q[1];
ry(1.8945753384890986) q[2];
cx q[1],q[2];
ry(-2.118078573341517) q[0];
ry(1.0120139938658248) q[1];
cx q[0],q[1];
ry(-1.4676425489885125) q[0];
ry(-2.383999298784375) q[1];
cx q[0],q[1];
ry(-0.11264606540873157) q[2];
ry(1.244500014714176) q[3];
cx q[2],q[3];
ry(-0.2651819353358613) q[2];
ry(0.42813224734081773) q[3];
cx q[2],q[3];
ry(-1.8864873866625826) q[0];
ry(-0.6355733898301748) q[2];
cx q[0],q[2];
ry(1.8938817534797767) q[0];
ry(2.4745515449551876) q[2];
cx q[0],q[2];
ry(0.9850113229817403) q[1];
ry(2.5263808686958997) q[3];
cx q[1],q[3];
ry(3.027505413401467) q[1];
ry(1.3890893769818984) q[3];
cx q[1],q[3];
ry(-2.60421545208827) q[0];
ry(2.7769041073458873) q[3];
cx q[0],q[3];
ry(-2.460080495778216) q[0];
ry(1.4639465819716597) q[3];
cx q[0],q[3];
ry(3.063763806507462) q[1];
ry(0.7817325158354288) q[2];
cx q[1],q[2];
ry(1.8589871818095043) q[1];
ry(-0.83605697432334) q[2];
cx q[1],q[2];
ry(0.9378156863703239) q[0];
ry(0.4795267515816839) q[1];
cx q[0],q[1];
ry(-1.6428994301841113) q[0];
ry(1.3536890819173586) q[1];
cx q[0],q[1];
ry(-1.2659713122337806) q[2];
ry(0.9078486978843693) q[3];
cx q[2],q[3];
ry(1.997159239260518) q[2];
ry(2.5773897011554143) q[3];
cx q[2],q[3];
ry(-2.334816166674318) q[0];
ry(1.7373175150607238) q[2];
cx q[0],q[2];
ry(-2.390757930516437) q[0];
ry(-2.098922748189347) q[2];
cx q[0],q[2];
ry(0.11141173701622104) q[1];
ry(0.39137851857255157) q[3];
cx q[1],q[3];
ry(-0.9086695496903853) q[1];
ry(2.498005335727159) q[3];
cx q[1],q[3];
ry(-0.4228739916310795) q[0];
ry(0.1377498890203528) q[3];
cx q[0],q[3];
ry(0.8850959985455791) q[0];
ry(-0.4388202699697423) q[3];
cx q[0],q[3];
ry(-2.9693273382305323) q[1];
ry(-0.049589266147273214) q[2];
cx q[1],q[2];
ry(1.1881899334670045) q[1];
ry(-0.7152056296296276) q[2];
cx q[1],q[2];
ry(-0.636490129704069) q[0];
ry(0.8425857867563895) q[1];
cx q[0],q[1];
ry(0.7874231038111943) q[0];
ry(-0.14330650488322613) q[1];
cx q[0],q[1];
ry(-0.535724154428807) q[2];
ry(2.5988199844056408) q[3];
cx q[2],q[3];
ry(-1.9022743951424772) q[2];
ry(1.546501143984191) q[3];
cx q[2],q[3];
ry(2.3884886886302397) q[0];
ry(-0.9495580559255831) q[2];
cx q[0],q[2];
ry(0.05177517603052733) q[0];
ry(1.434195497046118) q[2];
cx q[0],q[2];
ry(-2.862062905124734) q[1];
ry(-2.7210684512946948) q[3];
cx q[1],q[3];
ry(0.8801333597759718) q[1];
ry(-2.3874120986829146) q[3];
cx q[1],q[3];
ry(1.8878682380732381) q[0];
ry(3.004982237347825) q[3];
cx q[0],q[3];
ry(0.5508203258995729) q[0];
ry(0.5644364199502435) q[3];
cx q[0],q[3];
ry(0.06711319151990658) q[1];
ry(-0.7444469702506955) q[2];
cx q[1],q[2];
ry(-1.626886487144003) q[1];
ry(1.854551641550783) q[2];
cx q[1],q[2];
ry(0.9608825595541809) q[0];
ry(-3.083121634008394) q[1];
cx q[0],q[1];
ry(2.733121351388537) q[0];
ry(1.650400877399902) q[1];
cx q[0],q[1];
ry(-3.03260667876335) q[2];
ry(1.6147349665258919) q[3];
cx q[2],q[3];
ry(1.3958095566056754) q[2];
ry(-0.3993834891988853) q[3];
cx q[2],q[3];
ry(2.9234670273700516) q[0];
ry(-3.068057664003099) q[2];
cx q[0],q[2];
ry(1.646888197677466) q[0];
ry(2.5605284449118404) q[2];
cx q[0],q[2];
ry(-2.2358028485177415) q[1];
ry(-0.07819418979480834) q[3];
cx q[1],q[3];
ry(1.2215546176355836) q[1];
ry(1.9014240852138937) q[3];
cx q[1],q[3];
ry(2.690367767712345) q[0];
ry(-0.5694241545362928) q[3];
cx q[0],q[3];
ry(-0.7251014492250396) q[0];
ry(1.7988302271742975) q[3];
cx q[0],q[3];
ry(-0.41957092242539407) q[1];
ry(1.770630948757602) q[2];
cx q[1],q[2];
ry(-1.3498277256993247) q[1];
ry(-1.3321522182596497) q[2];
cx q[1],q[2];
ry(-2.7849177423236657) q[0];
ry(-2.2752188222699274) q[1];
cx q[0],q[1];
ry(-0.9906023819320415) q[0];
ry(-0.4401662805108816) q[1];
cx q[0],q[1];
ry(1.7503496607206093) q[2];
ry(1.6259704687103678) q[3];
cx q[2],q[3];
ry(2.1504527749307245) q[2];
ry(2.730904393040219) q[3];
cx q[2],q[3];
ry(-1.9778616882489086) q[0];
ry(-2.959897867786504) q[2];
cx q[0],q[2];
ry(0.37141514270567083) q[0];
ry(1.96633435085606) q[2];
cx q[0],q[2];
ry(2.196127233599474) q[1];
ry(-2.5758908695679184) q[3];
cx q[1],q[3];
ry(0.8040053362614925) q[1];
ry(-2.337111795526073) q[3];
cx q[1],q[3];
ry(0.3128674572833039) q[0];
ry(0.9290040294011375) q[3];
cx q[0],q[3];
ry(3.1406169566943047) q[0];
ry(2.615929298694264) q[3];
cx q[0],q[3];
ry(-2.9166938789937276) q[1];
ry(-1.059346765024575) q[2];
cx q[1],q[2];
ry(-0.755006567739486) q[1];
ry(-1.979506248197361) q[2];
cx q[1],q[2];
ry(-2.7403532157051713) q[0];
ry(1.2168811987461163) q[1];
cx q[0],q[1];
ry(-2.6222053575941233) q[0];
ry(3.0889727035904624) q[1];
cx q[0],q[1];
ry(0.11408203384328301) q[2];
ry(-0.29380571375059983) q[3];
cx q[2],q[3];
ry(-3.1179442605649266) q[2];
ry(-2.817000434943641) q[3];
cx q[2],q[3];
ry(-0.7401169180426034) q[0];
ry(0.02174302411361939) q[2];
cx q[0],q[2];
ry(-1.018040791544406) q[0];
ry(2.8934445648034184) q[2];
cx q[0],q[2];
ry(2.215683841416138) q[1];
ry(2.592679337444762) q[3];
cx q[1],q[3];
ry(-2.1541467181218485) q[1];
ry(-3.1398033341747844) q[3];
cx q[1],q[3];
ry(1.7550790491972097) q[0];
ry(2.8980403555193406) q[3];
cx q[0],q[3];
ry(-0.1807100825181197) q[0];
ry(1.5943966029312029) q[3];
cx q[0],q[3];
ry(0.7057190434582629) q[1];
ry(-2.203892789517199) q[2];
cx q[1],q[2];
ry(2.63338584318898) q[1];
ry(0.7662152711003055) q[2];
cx q[1],q[2];
ry(-2.4466759813853725) q[0];
ry(1.4213778801075325) q[1];
cx q[0],q[1];
ry(-0.5311366040007965) q[0];
ry(-2.072604131084849) q[1];
cx q[0],q[1];
ry(0.5777134983681229) q[2];
ry(2.946907665036914) q[3];
cx q[2],q[3];
ry(-1.644358858546374) q[2];
ry(-0.40764481230105465) q[3];
cx q[2],q[3];
ry(-0.21932550169242226) q[0];
ry(-3.0691458131643414) q[2];
cx q[0],q[2];
ry(2.481929759255022) q[0];
ry(-1.7878240800273075) q[2];
cx q[0],q[2];
ry(-1.452269515426964) q[1];
ry(1.7450809927611026) q[3];
cx q[1],q[3];
ry(-0.860654786259829) q[1];
ry(1.2838286420997858) q[3];
cx q[1],q[3];
ry(-1.5825115264621814) q[0];
ry(-2.664230400624823) q[3];
cx q[0],q[3];
ry(1.170718460070329) q[0];
ry(1.2847601070143604) q[3];
cx q[0],q[3];
ry(0.8786288723791511) q[1];
ry(0.22538098847929433) q[2];
cx q[1],q[2];
ry(-1.9158627196925355) q[1];
ry(-2.4007969888209946) q[2];
cx q[1],q[2];
ry(-1.9692814537589916) q[0];
ry(-0.10552360133719596) q[1];
cx q[0],q[1];
ry(2.7137056839758844) q[0];
ry(-0.9946514730579098) q[1];
cx q[0],q[1];
ry(-1.5709027465952372) q[2];
ry(-1.7374370835294028) q[3];
cx q[2],q[3];
ry(-1.201249953103524) q[2];
ry(-0.1601895663986257) q[3];
cx q[2],q[3];
ry(1.9497166478461805) q[0];
ry(-2.503450475999355) q[2];
cx q[0],q[2];
ry(1.1804479255232554) q[0];
ry(-0.14589543961734194) q[2];
cx q[0],q[2];
ry(0.8220611832692919) q[1];
ry(-0.439700998777395) q[3];
cx q[1],q[3];
ry(0.7837013154493899) q[1];
ry(2.312351584735508) q[3];
cx q[1],q[3];
ry(-0.5540876205207113) q[0];
ry(3.1251892101429077) q[3];
cx q[0],q[3];
ry(2.1159625925211474) q[0];
ry(1.9223434775103263) q[3];
cx q[0],q[3];
ry(-1.6581722937384806) q[1];
ry(-1.3392179968374824) q[2];
cx q[1],q[2];
ry(-2.1421777256267935) q[1];
ry(2.8153589891420605) q[2];
cx q[1],q[2];
ry(0.8308715102226598) q[0];
ry(1.2985267651655343) q[1];
cx q[0],q[1];
ry(2.56350492005319) q[0];
ry(-0.40639767099233465) q[1];
cx q[0],q[1];
ry(-2.906438585484589) q[2];
ry(2.2031608111220686) q[3];
cx q[2],q[3];
ry(1.811459741297118) q[2];
ry(-2.4203704391625522) q[3];
cx q[2],q[3];
ry(-1.0406068297419622) q[0];
ry(1.9463112957493411) q[2];
cx q[0],q[2];
ry(1.1928623179335478) q[0];
ry(0.4557859698645066) q[2];
cx q[0],q[2];
ry(-2.0172470896367525) q[1];
ry(2.5398883631836755) q[3];
cx q[1],q[3];
ry(-2.1811984045937844) q[1];
ry(-2.3678915255627757) q[3];
cx q[1],q[3];
ry(1.9936082252303269) q[0];
ry(-2.501207850462852) q[3];
cx q[0],q[3];
ry(1.7781603800008532) q[0];
ry(0.22198924271742085) q[3];
cx q[0],q[3];
ry(-1.138916989491967) q[1];
ry(0.2980183169235131) q[2];
cx q[1],q[2];
ry(2.2757326478095345) q[1];
ry(0.5398797680762515) q[2];
cx q[1],q[2];
ry(-2.7608329340041644) q[0];
ry(1.6187922919841473) q[1];
cx q[0],q[1];
ry(-0.1533856088391324) q[0];
ry(1.7334793614145427) q[1];
cx q[0],q[1];
ry(1.7258658978709152) q[2];
ry(0.629520273517902) q[3];
cx q[2],q[3];
ry(1.3727159306729115) q[2];
ry(-0.27927996127042687) q[3];
cx q[2],q[3];
ry(-2.019939043201875) q[0];
ry(-2.6197891724335256) q[2];
cx q[0],q[2];
ry(0.9164999274071134) q[0];
ry(-2.664183390687846) q[2];
cx q[0],q[2];
ry(2.448473445254904) q[1];
ry(-2.3058564013500495) q[3];
cx q[1],q[3];
ry(-3.092331573776794) q[1];
ry(-1.8269551746199397) q[3];
cx q[1],q[3];
ry(-2.4871515666371633) q[0];
ry(-1.3049135180431979) q[3];
cx q[0],q[3];
ry(-2.4819337280805316) q[0];
ry(0.9118877782858943) q[3];
cx q[0],q[3];
ry(0.8363974639430072) q[1];
ry(-3.020493209409227) q[2];
cx q[1],q[2];
ry(2.1836827087976407) q[1];
ry(1.3920560823979784) q[2];
cx q[1],q[2];
ry(-1.9186324251902143) q[0];
ry(-0.9572848753517667) q[1];
cx q[0],q[1];
ry(-0.28524503095523185) q[0];
ry(1.873909205156056) q[1];
cx q[0],q[1];
ry(1.960727159650648) q[2];
ry(-2.6289423434398604) q[3];
cx q[2],q[3];
ry(0.9837167860278805) q[2];
ry(0.6594155431120212) q[3];
cx q[2],q[3];
ry(2.9122728362227956) q[0];
ry(1.0568719858192028) q[2];
cx q[0],q[2];
ry(0.10160699709203945) q[0];
ry(-1.9192417938959752) q[2];
cx q[0],q[2];
ry(-1.1206628810670711) q[1];
ry(-1.7299555380043392) q[3];
cx q[1],q[3];
ry(0.4172302629700237) q[1];
ry(2.7099326004776487) q[3];
cx q[1],q[3];
ry(-2.192122898200366) q[0];
ry(2.939781884048637) q[3];
cx q[0],q[3];
ry(-2.525194042629434) q[0];
ry(1.154782309729427) q[3];
cx q[0],q[3];
ry(-0.8745035508467218) q[1];
ry(0.5481761327204865) q[2];
cx q[1],q[2];
ry(-2.2567074854487945) q[1];
ry(0.3100809444961836) q[2];
cx q[1],q[2];
ry(2.5538592351556604) q[0];
ry(1.237928848425179) q[1];
cx q[0],q[1];
ry(-0.3548412899436899) q[0];
ry(0.7337369754879273) q[1];
cx q[0],q[1];
ry(-2.0614509917832877) q[2];
ry(-1.8120895394035237) q[3];
cx q[2],q[3];
ry(0.37998773276227854) q[2];
ry(-0.27881915036809385) q[3];
cx q[2],q[3];
ry(-2.81374135323502) q[0];
ry(-2.0801703149119097) q[2];
cx q[0],q[2];
ry(0.3635489504564404) q[0];
ry(0.371821129933811) q[2];
cx q[0],q[2];
ry(-1.5019622285975291) q[1];
ry(-0.5751636733426633) q[3];
cx q[1],q[3];
ry(1.5963280866637701) q[1];
ry(1.1909885612745574) q[3];
cx q[1],q[3];
ry(1.442142840698859) q[0];
ry(2.242589822567168) q[3];
cx q[0],q[3];
ry(-0.7573083211906928) q[0];
ry(-0.7154041987130846) q[3];
cx q[0],q[3];
ry(-0.4211643924765554) q[1];
ry(2.869502602435058) q[2];
cx q[1],q[2];
ry(0.7067815295852311) q[1];
ry(1.118723187462086) q[2];
cx q[1],q[2];
ry(0.672843965650304) q[0];
ry(1.8823303846354356) q[1];
cx q[0],q[1];
ry(0.8184506226295376) q[0];
ry(2.6712436838797564) q[1];
cx q[0],q[1];
ry(2.8358175261226926) q[2];
ry(1.7249480904475423) q[3];
cx q[2],q[3];
ry(0.7220799680842188) q[2];
ry(0.008845492948770683) q[3];
cx q[2],q[3];
ry(0.11464197490372464) q[0];
ry(-0.6559606794501669) q[2];
cx q[0],q[2];
ry(1.5889740485257864) q[0];
ry(0.516900505005002) q[2];
cx q[0],q[2];
ry(0.3862782188942111) q[1];
ry(2.6763475347455072) q[3];
cx q[1],q[3];
ry(-1.3432805850158813) q[1];
ry(1.8723239463973005) q[3];
cx q[1],q[3];
ry(0.8636600779306853) q[0];
ry(0.6996984141240886) q[3];
cx q[0],q[3];
ry(-1.2436272307996816) q[0];
ry(1.902755989264352) q[3];
cx q[0],q[3];
ry(-0.4008547262424239) q[1];
ry(1.6769683798178168) q[2];
cx q[1],q[2];
ry(2.6553721148605547) q[1];
ry(-1.4175566144677714) q[2];
cx q[1],q[2];
ry(1.0295794079441487) q[0];
ry(-2.3867927598046212) q[1];
cx q[0],q[1];
ry(-2.3508575775954523) q[0];
ry(-2.23172963086341) q[1];
cx q[0],q[1];
ry(-2.84877254864618) q[2];
ry(2.3657342148750597) q[3];
cx q[2],q[3];
ry(0.3920144270918211) q[2];
ry(-1.172011428305483) q[3];
cx q[2],q[3];
ry(2.9371092315873906) q[0];
ry(1.1776478452740777) q[2];
cx q[0],q[2];
ry(0.4750541593246682) q[0];
ry(3.0163076250647793) q[2];
cx q[0],q[2];
ry(-2.0510771031054933) q[1];
ry(-3.1344539282971877) q[3];
cx q[1],q[3];
ry(-0.7381971725260695) q[1];
ry(2.584129189964685) q[3];
cx q[1],q[3];
ry(0.8653239816548162) q[0];
ry(-0.3319503573393048) q[3];
cx q[0],q[3];
ry(-2.0039856088367154) q[0];
ry(-0.9620451168634212) q[3];
cx q[0],q[3];
ry(-0.0169087765405513) q[1];
ry(-2.413700474675436) q[2];
cx q[1],q[2];
ry(2.6351598473931355) q[1];
ry(2.3274637367809032) q[2];
cx q[1],q[2];
ry(0.10078254844334555) q[0];
ry(-1.5472070291977484) q[1];
cx q[0],q[1];
ry(-0.64013476156879) q[0];
ry(-2.0874244888910978) q[1];
cx q[0],q[1];
ry(0.34943023823774916) q[2];
ry(-0.5684713382377581) q[3];
cx q[2],q[3];
ry(-0.36941401460671747) q[2];
ry(-2.7662240650613885) q[3];
cx q[2],q[3];
ry(1.0903159234342876) q[0];
ry(1.0218265017751338) q[2];
cx q[0],q[2];
ry(-1.0605683412639126) q[0];
ry(1.3011421863520773) q[2];
cx q[0],q[2];
ry(-0.21824243188218165) q[1];
ry(2.1006807319822838) q[3];
cx q[1],q[3];
ry(-1.468869124672107) q[1];
ry(-0.1622909376407814) q[3];
cx q[1],q[3];
ry(3.0011382021322155) q[0];
ry(1.112502370900195) q[3];
cx q[0],q[3];
ry(-1.170771704172127) q[0];
ry(-0.2806799826537733) q[3];
cx q[0],q[3];
ry(-1.0913900960635614) q[1];
ry(2.102549778391519) q[2];
cx q[1],q[2];
ry(-2.840008321427323) q[1];
ry(-2.254180663539736) q[2];
cx q[1],q[2];
ry(-1.0986216159540128) q[0];
ry(-2.403033333169445) q[1];
cx q[0],q[1];
ry(1.8146613319077067) q[0];
ry(-1.1043518988113366) q[1];
cx q[0],q[1];
ry(-2.4216353246268096) q[2];
ry(-1.5283767896713671) q[3];
cx q[2],q[3];
ry(-2.438632455712902) q[2];
ry(2.5936383054855554) q[3];
cx q[2],q[3];
ry(0.16708181513261688) q[0];
ry(2.34340015350032) q[2];
cx q[0],q[2];
ry(1.982230551733544) q[0];
ry(-0.7262557888782482) q[2];
cx q[0],q[2];
ry(0.9434312988034345) q[1];
ry(1.113320095181274) q[3];
cx q[1],q[3];
ry(2.1437793665587996) q[1];
ry(-0.3700428869486964) q[3];
cx q[1],q[3];
ry(3.102987992410229) q[0];
ry(-1.303175760438564) q[3];
cx q[0],q[3];
ry(2.383952654684938) q[0];
ry(2.285693378182471) q[3];
cx q[0],q[3];
ry(1.2792554631945672) q[1];
ry(-2.23029580097532) q[2];
cx q[1],q[2];
ry(1.9118489645616625) q[1];
ry(0.9782660024405899) q[2];
cx q[1],q[2];
ry(-2.0776238203437645) q[0];
ry(-2.668825354878338) q[1];
cx q[0],q[1];
ry(-0.9227986243737587) q[0];
ry(1.694401165838404) q[1];
cx q[0],q[1];
ry(-1.7665501441463718) q[2];
ry(1.8998035817000725) q[3];
cx q[2],q[3];
ry(-2.4159986733230516) q[2];
ry(-2.1521540326944733) q[3];
cx q[2],q[3];
ry(1.224280171386098) q[0];
ry(2.2961532209867146) q[2];
cx q[0],q[2];
ry(-0.2923500603661937) q[0];
ry(-2.7385759738911095) q[2];
cx q[0],q[2];
ry(2.0346990929572843) q[1];
ry(-2.4505340834025056) q[3];
cx q[1],q[3];
ry(3.055496689456022) q[1];
ry(0.3384284369532846) q[3];
cx q[1],q[3];
ry(1.0724678416158142) q[0];
ry(0.9687659689935327) q[3];
cx q[0],q[3];
ry(0.9512392687027904) q[0];
ry(-2.942613740743285) q[3];
cx q[0],q[3];
ry(-0.5134369272384453) q[1];
ry(-2.5405738128520032) q[2];
cx q[1],q[2];
ry(-1.7988581925968643) q[1];
ry(0.34510087444988274) q[2];
cx q[1],q[2];
ry(-0.30335244311652215) q[0];
ry(2.1794453689748634) q[1];
cx q[0],q[1];
ry(-1.528025145094749) q[0];
ry(1.4022468051246375) q[1];
cx q[0],q[1];
ry(-2.29164322389439) q[2];
ry(-2.9913430775818464) q[3];
cx q[2],q[3];
ry(-0.12796383511378906) q[2];
ry(-2.7953788421916164) q[3];
cx q[2],q[3];
ry(-2.168273673318709) q[0];
ry(1.9709543131730376) q[2];
cx q[0],q[2];
ry(-0.9007862343788775) q[0];
ry(1.983439102343616) q[2];
cx q[0],q[2];
ry(-0.4257435029700485) q[1];
ry(2.27626760772001) q[3];
cx q[1],q[3];
ry(-1.5274436885270737) q[1];
ry(2.2508816316329447) q[3];
cx q[1],q[3];
ry(1.6497235922824312) q[0];
ry(1.174215272484845) q[3];
cx q[0],q[3];
ry(-2.1759483202998373) q[0];
ry(-2.6894868560879694) q[3];
cx q[0],q[3];
ry(-1.4451506482707934) q[1];
ry(0.0055274199372390915) q[2];
cx q[1],q[2];
ry(2.678180235359308) q[1];
ry(3.084402109106427) q[2];
cx q[1],q[2];
ry(-1.9980895639875) q[0];
ry(2.997888169095897) q[1];
cx q[0],q[1];
ry(2.756338330014442) q[0];
ry(0.31387360887102306) q[1];
cx q[0],q[1];
ry(1.7345332410033345) q[2];
ry(-0.10543315686908539) q[3];
cx q[2],q[3];
ry(0.24276468300692858) q[2];
ry(0.38200956915719025) q[3];
cx q[2],q[3];
ry(1.5447776982358674) q[0];
ry(1.8789615069052585) q[2];
cx q[0],q[2];
ry(3.12954610392639) q[0];
ry(-0.4391672449996257) q[2];
cx q[0],q[2];
ry(2.2776135817968886) q[1];
ry(1.6227106026109501) q[3];
cx q[1],q[3];
ry(-0.7831114209670158) q[1];
ry(-2.3869368043553507) q[3];
cx q[1],q[3];
ry(1.700967248635048) q[0];
ry(0.48384178114059667) q[3];
cx q[0],q[3];
ry(0.1244195792285989) q[0];
ry(-2.101211870960686) q[3];
cx q[0],q[3];
ry(2.728100367155112) q[1];
ry(-2.796103070938857) q[2];
cx q[1],q[2];
ry(1.0704626944361273) q[1];
ry(1.0895157840125764) q[2];
cx q[1],q[2];
ry(-0.5632853733865238) q[0];
ry(2.1356412901648216) q[1];
cx q[0],q[1];
ry(1.7803271828566778) q[0];
ry(-2.8659693482351853) q[1];
cx q[0],q[1];
ry(2.8854958239168624) q[2];
ry(-2.9432445337921282) q[3];
cx q[2],q[3];
ry(-0.9027467979624689) q[2];
ry(0.8654006340892622) q[3];
cx q[2],q[3];
ry(1.664799854625267) q[0];
ry(0.9275014575403953) q[2];
cx q[0],q[2];
ry(-1.4464558436859898) q[0];
ry(1.141406773232672) q[2];
cx q[0],q[2];
ry(-0.7520453377798288) q[1];
ry(-1.8191060603989433) q[3];
cx q[1],q[3];
ry(-0.621712476001715) q[1];
ry(-1.2526673166779485) q[3];
cx q[1],q[3];
ry(0.7942607935130406) q[0];
ry(-0.5998348041406434) q[3];
cx q[0],q[3];
ry(-0.9348861017825819) q[0];
ry(-0.9111207615407801) q[3];
cx q[0],q[3];
ry(2.770880270565365) q[1];
ry(2.2331272413493695) q[2];
cx q[1],q[2];
ry(0.9762178154063564) q[1];
ry(-2.7403145541927936) q[2];
cx q[1],q[2];
ry(1.0327257394766998) q[0];
ry(0.8152634469339138) q[1];
cx q[0],q[1];
ry(0.5109028652608069) q[0];
ry(0.5923614020539052) q[1];
cx q[0],q[1];
ry(3.0882657230585875) q[2];
ry(1.634038751782426) q[3];
cx q[2],q[3];
ry(-0.3338733878328232) q[2];
ry(-0.8726451819872812) q[3];
cx q[2],q[3];
ry(-3.12329395520805) q[0];
ry(-2.9933940289110454) q[2];
cx q[0],q[2];
ry(1.3937466710425896) q[0];
ry(-0.9106481420962425) q[2];
cx q[0],q[2];
ry(2.731307064640606) q[1];
ry(-1.6073079414399685) q[3];
cx q[1],q[3];
ry(2.7016124199619993) q[1];
ry(0.10168070704520726) q[3];
cx q[1],q[3];
ry(1.7395258603640062) q[0];
ry(2.8092162846869386) q[3];
cx q[0],q[3];
ry(-1.83921111769134) q[0];
ry(2.2718784290735563) q[3];
cx q[0],q[3];
ry(1.5127523099706774) q[1];
ry(2.589691406111672) q[2];
cx q[1],q[2];
ry(0.2523141610228778) q[1];
ry(-0.9062714784631635) q[2];
cx q[1],q[2];
ry(1.4951439219249887) q[0];
ry(2.7976204247981875) q[1];
ry(1.1151888704762642) q[2];
ry(1.998572108453038) q[3];