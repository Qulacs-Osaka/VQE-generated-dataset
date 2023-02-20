OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.4658298684574865) q[0];
rz(-1.136169743718214) q[0];
ry(0.6211566130922844) q[1];
rz(-1.7692123555805548) q[1];
ry(-0.7995181482894633) q[2];
rz(-0.949972669808646) q[2];
ry(1.3310989082458224) q[3];
rz(-2.9351078068968555) q[3];
ry(-2.30873424716004) q[4];
rz(1.2599976950428413) q[4];
ry(-2.2205859262054837) q[5];
rz(-2.752050916788181) q[5];
ry(-0.8755085489025859) q[6];
rz(-2.2852120178840156) q[6];
ry(2.1325959137378487) q[7];
rz(2.9939448467927923) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.8158189702879386) q[0];
rz(-2.7177735408419355) q[0];
ry(0.25583068921835306) q[1];
rz(-0.6361017536407871) q[1];
ry(2.453707762970663) q[2];
rz(0.16349938699897415) q[2];
ry(-1.3581322817645651) q[3];
rz(1.9923315421150238) q[3];
ry(-0.7087712538654597) q[4];
rz(0.997808374521564) q[4];
ry(-0.4215147042829903) q[5];
rz(-0.9155627415056152) q[5];
ry(1.9299089051600964) q[6];
rz(-1.2893849895347893) q[6];
ry(1.6734190900038006) q[7];
rz(1.312549619136682) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.56364998974369) q[0];
rz(-0.10800624595821329) q[0];
ry(-2.65893176950655) q[1];
rz(1.3201150092062162) q[1];
ry(-2.4437878774034827) q[2];
rz(-0.8747430795767397) q[2];
ry(-2.280564405155732) q[3];
rz(0.8828465723854108) q[3];
ry(0.4091226297566362) q[4];
rz(-1.8569383870712928) q[4];
ry(-1.5638245605462924) q[5];
rz(2.3385196688792846) q[5];
ry(-0.09153856850534581) q[6];
rz(-2.5694607373381246) q[6];
ry(-2.4332369225204724) q[7];
rz(-0.6297097425054516) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.0119540919981425) q[0];
rz(-1.192797294941708) q[0];
ry(-1.3387566966577884) q[1];
rz(-2.158091244379133) q[1];
ry(-1.9957441117202173) q[2];
rz(1.365026672934234) q[2];
ry(0.5057689805515843) q[3];
rz(-2.5398323166288046) q[3];
ry(-1.019547662130404) q[4];
rz(1.7316013221936766) q[4];
ry(-1.0591551096985272) q[5];
rz(-2.3159079525035735) q[5];
ry(-2.2360769584707847) q[6];
rz(-1.1431350306534984) q[6];
ry(-1.4406374283567884) q[7];
rz(-1.0216855522084807) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.575150840843806) q[0];
rz(-1.7703198288491633) q[0];
ry(1.5441157448862501) q[1];
rz(0.09285038971672224) q[1];
ry(0.3327511998250981) q[2];
rz(1.5933493367705234) q[2];
ry(-1.5450502373800028) q[3];
rz(2.23279441689162) q[3];
ry(-3.0940609926773823) q[4];
rz(-0.6909323511708161) q[4];
ry(1.29789733047704) q[5];
rz(-2.6717033938267214) q[5];
ry(-2.4142574008648245) q[6];
rz(-0.0915656288240827) q[6];
ry(0.7154950951610352) q[7];
rz(2.72013222872385) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.9212702775297514) q[0];
rz(-1.6949702428282283) q[0];
ry(-1.2088536524519515) q[1];
rz(2.6691731155533236) q[1];
ry(-2.5726842623856085) q[2];
rz(-0.0874580763475625) q[2];
ry(-2.212976922116393) q[3];
rz(2.850061070461665) q[3];
ry(0.12488137900960375) q[4];
rz(2.231173025900448) q[4];
ry(-1.2322598331400627) q[5];
rz(1.707675560742102) q[5];
ry(2.2813225618759994) q[6];
rz(2.8737661706455153) q[6];
ry(0.5571921021147737) q[7];
rz(-2.9023986794652634) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.8139032093320426) q[0];
rz(-1.844675989958014) q[0];
ry(-0.9303112313764492) q[1];
rz(-1.9391818196542994) q[1];
ry(1.0321941571588933) q[2];
rz(-2.0412947181613905) q[2];
ry(1.7608447647825425) q[3];
rz(-0.6451132630340667) q[3];
ry(1.1189961621524596) q[4];
rz(-0.3953476178028561) q[4];
ry(0.6157014926071823) q[5];
rz(-0.12909841755629525) q[5];
ry(-0.4594815932901689) q[6];
rz(2.7837525477994105) q[6];
ry(1.2568371997127517) q[7];
rz(2.9183727873194094) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.1122767363924213) q[0];
rz(2.6134308535749122) q[0];
ry(-0.7118606876594652) q[1];
rz(0.5363002997157763) q[1];
ry(-1.2568034395761118) q[2];
rz(-1.2683298803617324) q[2];
ry(-0.5697085690388519) q[3];
rz(1.5432792447099206) q[3];
ry(-0.49731372627726866) q[4];
rz(1.6971860926461177) q[4];
ry(2.8513430358509932) q[5];
rz(2.2764225456960423) q[5];
ry(2.2369998684616066) q[6];
rz(-1.7038903158481284) q[6];
ry(-0.4075315969514542) q[7];
rz(-0.33981303917848393) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.0369467357586077) q[0];
rz(-2.228593996794459) q[0];
ry(-2.574255779539144) q[1];
rz(-1.1212911910175807) q[1];
ry(0.24547725193029282) q[2];
rz(-0.4239802441022409) q[2];
ry(0.3941164389754149) q[3];
rz(-0.49163233070573137) q[3];
ry(2.3552014101068903) q[4];
rz(-2.6770394085253244) q[4];
ry(-1.1712869159186297) q[5];
rz(2.549084703164583) q[5];
ry(2.6416860967271862) q[6];
rz(-0.7079759624515657) q[6];
ry(0.1648941522088275) q[7];
rz(1.3926524711236512) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.134217512762465) q[0];
rz(-2.8920145328965243) q[0];
ry(0.2571208725512122) q[1];
rz(1.6825168092399905) q[1];
ry(-0.39269164412719154) q[2];
rz(-2.8905585510004927) q[2];
ry(1.398613152425682) q[3];
rz(-3.1234834391321935) q[3];
ry(2.8845715810821106) q[4];
rz(0.9926442130316794) q[4];
ry(2.648565551906176) q[5];
rz(-1.108296291209195) q[5];
ry(-0.8448182065539003) q[6];
rz(1.3373453645802797) q[6];
ry(-1.1154000685001353) q[7];
rz(-1.6484642733338433) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.5061377910416835) q[0];
rz(1.285310244748758) q[0];
ry(-2.500084789671889) q[1];
rz(0.10001115287201096) q[1];
ry(0.9790522905452326) q[2];
rz(1.2619536957958282) q[2];
ry(2.4412266195617436) q[3];
rz(0.6263000451136556) q[3];
ry(1.3758207173397674) q[4];
rz(0.0077909080245322756) q[4];
ry(-2.8561467615881884) q[5];
rz(-1.0723429703326928) q[5];
ry(0.7888930692551787) q[6];
rz(-0.48601904627823955) q[6];
ry(-0.8415925589165781) q[7];
rz(1.5847128856427166) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.5655932550167027) q[0];
rz(2.9241219642961136) q[0];
ry(-1.7770307428266432) q[1];
rz(2.231412932178179) q[1];
ry(-0.28701804441390966) q[2];
rz(1.5009194413315396) q[2];
ry(-1.715517373201824) q[3];
rz(0.627400955528607) q[3];
ry(-2.105041299554193) q[4];
rz(-1.7767931817145337) q[4];
ry(-2.0408400445238706) q[5];
rz(-1.1808533837696986) q[5];
ry(-0.5851334516093569) q[6];
rz(0.842864292387885) q[6];
ry(-0.5529318715457262) q[7];
rz(0.4222462009587602) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.9324276916386247) q[0];
rz(-1.8363949522842218) q[0];
ry(-0.5635600767319575) q[1];
rz(-1.2515082832411861) q[1];
ry(-1.8593620755799554) q[2];
rz(2.13914536248086) q[2];
ry(0.4470870595961056) q[3];
rz(0.9011325464362684) q[3];
ry(1.665897187114744) q[4];
rz(1.2655766046929384) q[4];
ry(-2.019518277906443) q[5];
rz(0.9962817266301576) q[5];
ry(-0.7668584236372746) q[6];
rz(2.8397176866685743) q[6];
ry(-0.5360584838097904) q[7];
rz(-2.9042574633429434) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.072154377065994) q[0];
rz(-0.23188042343105847) q[0];
ry(2.1649431213915893) q[1];
rz(-1.1317060230916447) q[1];
ry(-0.7889838092275597) q[2];
rz(0.3428844987378161) q[2];
ry(0.6542473491048614) q[3];
rz(-1.6164669661359836) q[3];
ry(-0.2760838929885359) q[4];
rz(-0.9899554248722895) q[4];
ry(-2.047360665072318) q[5];
rz(-2.4820258237388533) q[5];
ry(0.9153677629389475) q[6];
rz(1.0026168468065808) q[6];
ry(-0.23567730762375041) q[7];
rz(-2.649925696609364) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.8439604702340935) q[0];
rz(-1.5927210225602462) q[0];
ry(0.6027419016306366) q[1];
rz(1.14788888051028) q[1];
ry(1.847523384829402) q[2];
rz(-1.3486354323214942) q[2];
ry(0.6864371815631722) q[3];
rz(0.17479822144114507) q[3];
ry(-0.27081207481332736) q[4];
rz(2.9904415292125175) q[4];
ry(1.8435362073450317) q[5];
rz(-1.2154113019873147) q[5];
ry(1.29890042488484) q[6];
rz(2.6274682567517624) q[6];
ry(0.7221012436816405) q[7];
rz(0.9127295920370716) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.7970221762906577) q[0];
rz(-0.6577618002891478) q[0];
ry(2.163903305262439) q[1];
rz(-0.6843708157794572) q[1];
ry(-2.767296677642898) q[2];
rz(-1.0093795095421818) q[2];
ry(-2.5167002487978114) q[3];
rz(-2.8866068677235512) q[3];
ry(0.05473767102515567) q[4];
rz(-0.15551572566500035) q[4];
ry(3.032008262793336) q[5];
rz(2.7554446793233285) q[5];
ry(2.0298943349586978) q[6];
rz(-1.2423898135784264) q[6];
ry(0.4102833696470005) q[7];
rz(-2.0690372047010333) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.1188546825176955) q[0];
rz(-1.0597518800752812) q[0];
ry(-1.8244905710665904) q[1];
rz(1.3055306719655357) q[1];
ry(2.6654005086408943) q[2];
rz(-1.0172338350865757) q[2];
ry(-1.0075253188378537) q[3];
rz(-2.108802759319648) q[3];
ry(-0.4001563263356003) q[4];
rz(-2.5945882753769847) q[4];
ry(-0.2730464105907586) q[5];
rz(3.044541721890372) q[5];
ry(-1.8862693858117832) q[6];
rz(1.7617096027183585) q[6];
ry(-2.26266340434576) q[7];
rz(0.24007596991331326) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.522246658572512) q[0];
rz(-0.3792563576637021) q[0];
ry(1.1432869060863222) q[1];
rz(-2.355663035457616) q[1];
ry(0.14211777672967418) q[2];
rz(0.08837836585907172) q[2];
ry(1.5815749295127735) q[3];
rz(2.1164819387982634) q[3];
ry(-0.5348458248092117) q[4];
rz(-3.139951125507135) q[4];
ry(-0.5190221054292254) q[5];
rz(-1.7987577423795613) q[5];
ry(1.8677280590707455) q[6];
rz(2.8611553517823567) q[6];
ry(-0.26387453460792365) q[7];
rz(-1.720344925642194) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.7777335683519766) q[0];
rz(-2.9542163082201847) q[0];
ry(-0.7169060222161308) q[1];
rz(-0.62018345037601) q[1];
ry(-2.1631509342794075) q[2];
rz(1.733800657897899) q[2];
ry(-1.3415792833453704) q[3];
rz(-1.691420742459153) q[3];
ry(0.9635236178923737) q[4];
rz(2.0694548311311047) q[4];
ry(1.4289877008088148) q[5];
rz(-2.2717905832464504) q[5];
ry(-2.699268370989834) q[6];
rz(-1.1700565853884195) q[6];
ry(-1.7113831113807727) q[7];
rz(-3.072304740231532) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.2928369199342424) q[0];
rz(-2.521510506639375) q[0];
ry(0.835426611261731) q[1];
rz(-2.0798606989276625) q[1];
ry(-2.2015549140117328) q[2];
rz(-3.122010438781994) q[2];
ry(0.3539557566651741) q[3];
rz(2.774901215630453) q[3];
ry(2.4205584673228815) q[4];
rz(-0.7245577662914551) q[4];
ry(-1.3293058279552892) q[5];
rz(1.4903089267005944) q[5];
ry(0.8294619493826527) q[6];
rz(1.0658358834248238) q[6];
ry(1.6722019214855541) q[7];
rz(0.32923538650268025) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.2534027583787872) q[0];
rz(-1.327311973310838) q[0];
ry(-1.7955707775622989) q[1];
rz(2.590402763035282) q[1];
ry(0.4661557105927434) q[2];
rz(1.4653331021386493) q[2];
ry(-1.9549692128262048) q[3];
rz(-0.8506534404187414) q[3];
ry(-1.6445860240647383) q[4];
rz(-0.8386744069618307) q[4];
ry(2.160803966900591) q[5];
rz(-2.8584958242003835) q[5];
ry(0.41536646661670656) q[6];
rz(-2.911228835301652) q[6];
ry(0.17914396769736207) q[7];
rz(-0.19637083622675452) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.9397380861126514) q[0];
rz(-2.7198429240828084) q[0];
ry(-3.0723381031135277) q[1];
rz(1.4694515909136596) q[1];
ry(-1.6500668921715818) q[2];
rz(-3.0205905028406876) q[2];
ry(-3.0960004510528787) q[3];
rz(-0.9032023022154467) q[3];
ry(-0.8775862363941169) q[4];
rz(0.19375313014026768) q[4];
ry(1.8267113876568652) q[5];
rz(0.4509972509451605) q[5];
ry(2.0756022800525447) q[6];
rz(2.650837172298488) q[6];
ry(0.5707507515511228) q[7];
rz(-1.747077097944337) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.9445144185457721) q[0];
rz(1.1045800298517596) q[0];
ry(2.6463868823094288) q[1];
rz(0.6761539751358568) q[1];
ry(0.6752698853330408) q[2];
rz(2.6499781721594853) q[2];
ry(1.336450220898067) q[3];
rz(-1.6663445385045124) q[3];
ry(0.6974994875958437) q[4];
rz(2.7372090068102226) q[4];
ry(-0.13839842554729787) q[5];
rz(-1.119335063833181) q[5];
ry(-0.6756309186849503) q[6];
rz(-0.5093919731951395) q[6];
ry(0.8190633401164655) q[7];
rz(1.8797277164366877) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.438872753501061) q[0];
rz(2.349498268552088) q[0];
ry(0.6598140781111264) q[1];
rz(-0.2501915587583188) q[1];
ry(-2.729889458374772) q[2];
rz(-2.7690920488824236) q[2];
ry(-0.13315810353195137) q[3];
rz(-2.0997357851830616) q[3];
ry(1.981765076531838) q[4];
rz(1.7930800843625727) q[4];
ry(-2.359990454479676) q[5];
rz(1.5177443033511295) q[5];
ry(1.3195576842470995) q[6];
rz(0.7265262855605656) q[6];
ry(2.407377890779501) q[7];
rz(-2.7204839464010853) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.9479215054132) q[0];
rz(-2.1985943974208384) q[0];
ry(2.1883932350637822) q[1];
rz(-2.892328887318689) q[1];
ry(-2.931576185169652) q[2];
rz(1.2667331263815633) q[2];
ry(-0.7610358682792029) q[3];
rz(-0.5268186861649635) q[3];
ry(-1.8447874747447612) q[4];
rz(-3.0444981975864973) q[4];
ry(0.11641023865933064) q[5];
rz(0.5251339793381802) q[5];
ry(-2.239009863531785) q[6];
rz(-0.41357448568505184) q[6];
ry(0.9232513722459048) q[7];
rz(-2.2715707765806967) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.0417815297079507) q[0];
rz(-1.3742544316411114) q[0];
ry(-0.5412159239793536) q[1];
rz(-2.1275548107879656) q[1];
ry(-0.7287374790141774) q[2];
rz(0.8084610237771629) q[2];
ry(-1.8635126539114235) q[3];
rz(2.4455116372660326) q[3];
ry(-2.420089090812666) q[4];
rz(-0.4693084040722313) q[4];
ry(0.607619372945952) q[5];
rz(1.2879676479732016) q[5];
ry(1.3512464973929879) q[6];
rz(2.7325828053865955) q[6];
ry(-2.5952579881637288) q[7];
rz(2.9316212124711494) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.054079716664636) q[0];
rz(-0.4113673250589027) q[0];
ry(-1.985798692544713) q[1];
rz(-1.8717566011577433) q[1];
ry(-2.368253573193675) q[2];
rz(-0.27652173931269636) q[2];
ry(2.0327898208614013) q[3];
rz(1.6062011440203285) q[3];
ry(0.40401381747436904) q[4];
rz(-1.2745537638394582) q[4];
ry(2.504736659208435) q[5];
rz(2.3720220485282533) q[5];
ry(0.38396524012165134) q[6];
rz(0.08400165192596099) q[6];
ry(-1.7193508644174926) q[7];
rz(2.4800389797143696) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.1056350772512964) q[0];
rz(2.718339724457741) q[0];
ry(-0.1670949299034259) q[1];
rz(-0.05496359751349101) q[1];
ry(0.8701812959604078) q[2];
rz(1.765428267752866) q[2];
ry(2.285167496658458) q[3];
rz(2.782100067991166) q[3];
ry(-0.5154263224617219) q[4];
rz(0.19960852914420887) q[4];
ry(1.7073009192241813) q[5];
rz(2.572643604728922) q[5];
ry(-2.332993294725178) q[6];
rz(-2.6857419947519374) q[6];
ry(0.15750143446470374) q[7];
rz(-1.3155536703236885) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-3.1370850145364026) q[0];
rz(0.3976847690467376) q[0];
ry(1.9306656296247084) q[1];
rz(1.4943009299053576) q[1];
ry(-1.1706437598028832) q[2];
rz(-1.6880965747376258) q[2];
ry(-0.48678190085373707) q[3];
rz(2.0270259260616683) q[3];
ry(-1.1617908670136874) q[4];
rz(-2.405007923144586) q[4];
ry(2.3505793109963835) q[5];
rz(-2.8210754578092114) q[5];
ry(0.6795611970410005) q[6];
rz(-0.6572545577623234) q[6];
ry(1.1812833264753158) q[7];
rz(2.676519976819713) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.4752627404045542) q[0];
rz(-2.103009221598968) q[0];
ry(-2.4756358323308723) q[1];
rz(-2.805220536195056) q[1];
ry(0.023888532203615767) q[2];
rz(-1.0521646703286394) q[2];
ry(1.5731130426403028) q[3];
rz(3.078482189076601) q[3];
ry(-2.847758554532075) q[4];
rz(0.07402763466291429) q[4];
ry(1.7865995232206053) q[5];
rz(-2.167567675179928) q[5];
ry(-0.9118251048132575) q[6];
rz(0.3820066641801425) q[6];
ry(-1.584315743379978) q[7];
rz(-0.03586975140510785) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.3166815763278326) q[0];
rz(-0.6431737230831758) q[0];
ry(-0.7835909410350008) q[1];
rz(0.005824424726257528) q[1];
ry(1.4424749406998962) q[2];
rz(0.09868814842681649) q[2];
ry(2.614008049483421) q[3];
rz(-1.884280254491872) q[3];
ry(-0.006781829715815313) q[4];
rz(0.9866729785215913) q[4];
ry(2.2812993305889258) q[5];
rz(2.3446568794651794) q[5];
ry(-3.1253365031224836) q[6];
rz(-3.0687558220227205) q[6];
ry(-1.6438332779276355) q[7];
rz(-2.589299336952654) q[7];