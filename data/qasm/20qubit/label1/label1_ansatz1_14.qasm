OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.2272297761727018) q[0];
rz(2.4188348067583116) q[0];
ry(-2.9478211261536234) q[1];
rz(2.3064949809768907) q[1];
ry(-2.9539828493936158) q[2];
rz(-1.9556055396960756) q[2];
ry(-1.8474224760608697) q[3];
rz(-2.119453796603744) q[3];
ry(2.2317862632829284) q[4];
rz(-0.448574195928661) q[4];
ry(-1.7462221635557196) q[5];
rz(3.034920860250128) q[5];
ry(-0.12037504362607276) q[6];
rz(-1.87553159538036) q[6];
ry(-0.04606908297847009) q[7];
rz(2.8007251404025526) q[7];
ry(-3.1309767862728832) q[8];
rz(-2.6329946530121258) q[8];
ry(1.4199944260148003) q[9];
rz(-0.6786624370570229) q[9];
ry(1.3270569783391641) q[10];
rz(2.4006370405809045) q[10];
ry(0.2960246574868453) q[11];
rz(1.4560385816018864) q[11];
ry(-3.1055563714449876) q[12];
rz(-0.8689664330956245) q[12];
ry(-1.2767586086586877) q[13];
rz(2.142314495892771) q[13];
ry(-1.317946803158477) q[14];
rz(2.0314488404830184) q[14];
ry(-0.2840196733368802) q[15];
rz(0.6900273646943438) q[15];
ry(-2.358593461660416) q[16];
rz(2.3346126302307195) q[16];
ry(1.0545700409875614) q[17];
rz(-0.20244409349959636) q[17];
ry(-2.653875150376718) q[18];
rz(-2.4204509998565498) q[18];
ry(0.1302248589906636) q[19];
rz(-1.066124960839586) q[19];
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
ry(-0.6878125343749445) q[0];
rz(-0.629071749725842) q[0];
ry(-0.32937993331009174) q[1];
rz(-3.0417314913880387) q[1];
ry(1.2257819259039984) q[2];
rz(0.907004970486054) q[2];
ry(0.27118924711419073) q[3];
rz(0.7472290823280154) q[3];
ry(-3.1249429975261256) q[4];
rz(2.0983765941878296) q[4];
ry(-1.1696117223930746) q[5];
rz(0.03502357799450481) q[5];
ry(0.038365417404876645) q[6];
rz(1.4218362610425137) q[6];
ry(0.024349360636287898) q[7];
rz(-1.7375803165461328) q[7];
ry(1.2669427136458882) q[8];
rz(3.114383705123681) q[8];
ry(-1.0036690441816827) q[9];
rz(-0.2626909021123673) q[9];
ry(2.763629922556436) q[10];
rz(2.8921220213595253) q[10];
ry(1.1936579074851446) q[11];
rz(-2.9907823372565923) q[11];
ry(0.055974809024737775) q[12];
rz(-2.717298738260054) q[12];
ry(-1.4784071846829168) q[13];
rz(0.17000097541898385) q[13];
ry(-2.651143923770298) q[14];
rz(-0.3584029561764578) q[14];
ry(2.370124762462497) q[15];
rz(2.9406739646705833) q[15];
ry(-0.03383972395505964) q[16];
rz(2.7344594098349524) q[16];
ry(-3.0988658218389826) q[17];
rz(2.58770738381326) q[17];
ry(-0.25148418387638394) q[18];
rz(-2.4627821433707213) q[18];
ry(-1.274043717783143) q[19];
rz(-1.3335747667784181) q[19];
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
ry(-1.1903796281724899) q[0];
rz(0.14742190575621855) q[0];
ry(-3.121545545546314) q[1];
rz(0.5986822317863307) q[1];
ry(2.6037699607216345) q[2];
rz(-2.3450750031732723) q[2];
ry(2.7541672852720227) q[3];
rz(1.9492100354726147) q[3];
ry(-1.6851748318696305) q[4];
rz(0.2126865901360544) q[4];
ry(-1.3734132314347232) q[5];
rz(-3.077936449620231) q[5];
ry(3.1220466250816052) q[6];
rz(-2.817700607803489) q[6];
ry(-2.0027014079255325) q[7];
rz(0.05815159928111146) q[7];
ry(0.9232817765740364) q[8];
rz(-2.965480351734742) q[8];
ry(3.1297383911048513) q[9];
rz(1.332929461174417) q[9];
ry(0.4573824011473864) q[10];
rz(0.6852081641557142) q[10];
ry(-0.5230058318489349) q[11];
rz(0.46275308231978496) q[11];
ry(-1.5435187395565795) q[12];
rz(0.9327649865686647) q[12];
ry(-1.775403712096658) q[13];
rz(-1.4207633385535894) q[13];
ry(-0.26869389816660405) q[14];
rz(-2.255354889598541) q[14];
ry(-2.852245608163362) q[15];
rz(-0.4212146825907697) q[15];
ry(-2.7344541405552913) q[16];
rz(0.9059546103095526) q[16];
ry(-2.121879077802779) q[17];
rz(2.2276917286651194) q[17];
ry(-0.035680092462655466) q[18];
rz(2.9218487892363068) q[18];
ry(0.4804596834667188) q[19];
rz(0.8191126328651694) q[19];
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
ry(2.6120134663170345) q[0];
rz(1.3402577952291486) q[0];
ry(0.25463993373853056) q[1];
rz(1.9917781874981708) q[1];
ry(1.5613145682943443) q[2];
rz(0.01256407432434603) q[2];
ry(3.1312271960000895) q[3];
rz(-0.838641508551758) q[3];
ry(-0.12414113488019032) q[4];
rz(-0.4290895581280587) q[4];
ry(0.4242228713429616) q[5];
rz(1.6128532427177458) q[5];
ry(1.5086372806874822) q[6];
rz(-1.5897218878449773) q[6];
ry(-0.734506148324888) q[7];
rz(-2.783727583027396) q[7];
ry(-0.0951362382822188) q[8];
rz(2.9346999552119826) q[8];
ry(-0.17017460737597467) q[9];
rz(0.23122637099239096) q[9];
ry(0.4922627251563673) q[10];
rz(2.3554452493520395) q[10];
ry(-1.5156673603729056) q[11];
rz(-0.05240277112890834) q[11];
ry(3.1021974646235893) q[12];
rz(2.912449306176834) q[12];
ry(-1.1240382878884856) q[13];
rz(2.3858462587756577) q[13];
ry(-1.3769511293387477) q[14];
rz(1.8205375908043815) q[14];
ry(0.8342029836514646) q[15];
rz(-1.8789356970417275) q[15];
ry(0.1863330997869003) q[16];
rz(-1.4285907825074704) q[16];
ry(0.23587415522036373) q[17];
rz(1.1605583985869625) q[17];
ry(0.9841947066086298) q[18];
rz(-0.587874598440336) q[18];
ry(2.941279296300008) q[19];
rz(-2.102342388912696) q[19];
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
ry(-1.295560924267508) q[0];
rz(0.37189965427945004) q[0];
ry(-0.44048265484551) q[1];
rz(-0.6957289920894009) q[1];
ry(-2.7678195585961856) q[2];
rz(-3.0976744855506344) q[2];
ry(0.8689524857139093) q[3];
rz(-2.249161699264593) q[3];
ry(1.4974015097634874) q[4];
rz(-1.8401456716185827) q[4];
ry(0.0038943799332676462) q[5];
rz(-1.4683240124072887) q[5];
ry(-0.2766207019643139) q[6];
rz(-1.6577645264285124) q[6];
ry(0.9933208796664592) q[7];
rz(-3.1408073336235733) q[7];
ry(-0.32936027133090207) q[8];
rz(2.927754509306576) q[8];
ry(2.994747878673145) q[9];
rz(-1.3611084788315255) q[9];
ry(-1.5887643285646318) q[10];
rz(-0.17687781466248162) q[10];
ry(-1.2765928639851536) q[11];
rz(0.5326985793428158) q[11];
ry(3.0777911103178064) q[12];
rz(2.880853674588346) q[12];
ry(1.6542826565157884) q[13];
rz(-0.4755039697261225) q[13];
ry(-0.7655453812647642) q[14];
rz(1.1332498888425888) q[14];
ry(0.7315136325609056) q[15];
rz(3.089093495366264) q[15];
ry(2.5558373054729375) q[16];
rz(1.0606608706153309) q[16];
ry(2.9699779080130706) q[17];
rz(-2.9548299762023107) q[17];
ry(0.05215823118335816) q[18];
rz(-1.2055202229065802) q[18];
ry(2.2718910529418626) q[19];
rz(-1.5281563182111473) q[19];
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
ry(1.0326606887239205) q[0];
rz(3.0505532354031066) q[0];
ry(1.9780204434916742) q[1];
rz(-0.9310776576477736) q[1];
ry(2.0703770408574096) q[2];
rz(-0.9447273713891473) q[2];
ry(1.2057787692900503) q[3];
rz(-2.516537509058946) q[3];
ry(0.05461062089805946) q[4];
rz(-0.46598427183330754) q[4];
ry(1.8504370913749115) q[5];
rz(-1.3820105621698247) q[5];
ry(2.8994921232755075) q[6];
rz(2.375301731899146) q[6];
ry(-2.9362993114181766) q[7];
rz(-0.13361592106109044) q[7];
ry(-3.1232799265532636) q[8];
rz(-1.7622905005092169) q[8];
ry(-1.534725371571394) q[9];
rz(-0.7636639596310486) q[9];
ry(2.919463075673896) q[10];
rz(-1.360770915447799) q[10];
ry(0.0007246058165058795) q[11];
rz(-0.4948906451316182) q[11];
ry(-0.8228268244106349) q[12];
rz(1.413997397102243) q[12];
ry(-1.7145672103626974) q[13];
rz(-1.7456666292061866) q[13];
ry(-0.13140355761987355) q[14];
rz(1.445372880823351) q[14];
ry(2.2917897484985508) q[15];
rz(-3.126387968513697) q[15];
ry(0.003431176094064625) q[16];
rz(1.76123389669165) q[16];
ry(-2.303028484729853) q[17];
rz(-0.4190340036686971) q[17];
ry(-1.4412572591556492) q[18];
rz(-1.8336926999613783) q[18];
ry(1.6474740614917278) q[19];
rz(-1.3887293684239195) q[19];
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
ry(2.1648544848179014) q[0];
rz(2.6501183077015504) q[0];
ry(3.047950887585226) q[1];
rz(1.009591221508716) q[1];
ry(3.106777707447262) q[2];
rz(-0.8058270346125909) q[2];
ry(-3.121398743867953) q[3];
rz(-2.268462944247471) q[3];
ry(1.7307184465304921) q[4];
rz(-0.16875717029648651) q[4];
ry(2.870070711429949) q[5];
rz(1.211300284269349) q[5];
ry(1.1141052917349885) q[6];
rz(-3.076869582151785) q[6];
ry(0.868199893360666) q[7];
rz(1.5177588703741796) q[7];
ry(-1.5733808355695285) q[8];
rz(2.772446984938214) q[8];
ry(-3.132528626396227) q[9];
rz(2.206846206822921) q[9];
ry(0.02680376091994502) q[10];
rz(-0.12878504927369505) q[10];
ry(2.8964412391538916) q[11];
rz(-2.549444058069424) q[11];
ry(-2.953288791514642) q[12];
rz(-0.3452220008176647) q[12];
ry(1.392370442686902) q[13];
rz(2.47797550277937) q[13];
ry(-3.056462420717311) q[14];
rz(1.1153184943192704) q[14];
ry(-0.7819928679052426) q[15];
rz(-0.4905342627440482) q[15];
ry(-2.9205549518526808) q[16];
rz(-0.4547496906776764) q[16];
ry(1.0702516287005919) q[17];
rz(-0.346754050512395) q[17];
ry(0.16303467532380234) q[18];
rz(-0.41414000189840117) q[18];
ry(0.1574597148277689) q[19];
rz(3.095042473512295) q[19];
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
ry(-1.5199210929435036) q[0];
rz(-0.13467347405881736) q[0];
ry(0.27974036011339054) q[1];
rz(0.487531727126818) q[1];
ry(-1.6063196900430263) q[2];
rz(1.163004991651741) q[2];
ry(-1.9189984854054698) q[3];
rz(-1.3814855555440326) q[3];
ry(-0.06106538788692095) q[4];
rz(-3.028010786406049) q[4];
ry(-0.0029069080724921648) q[5];
rz(1.0118505571137437) q[5];
ry(0.9864442641191586) q[6];
rz(-0.9022732200012397) q[6];
ry(-1.5717252344527983) q[7];
rz(0.18011589614012546) q[7];
ry(-0.12244265315269143) q[8];
rz(2.388586770904138) q[8];
ry(-0.6231615755871248) q[9];
rz(0.6946253751751321) q[9];
ry(-0.730123256955948) q[10];
rz(-2.021118288325539) q[10];
ry(-0.2987604173487001) q[11];
rz(0.7919545908543926) q[11];
ry(2.9090196592913475) q[12];
rz(-2.6209898086970247) q[12];
ry(-2.709025566138356) q[13];
rz(2.335096442540979) q[13];
ry(-1.6245824964883597) q[14];
rz(-1.0224838010991018) q[14];
ry(-2.3249031658813606) q[15];
rz(2.8647029617776623) q[15];
ry(2.984985914128788) q[16];
rz(-1.8740551003799735) q[16];
ry(-2.103297853390251) q[17];
rz(-1.480510247680899) q[17];
ry(2.224082288874175) q[18];
rz(2.095620535970223) q[18];
ry(0.49865916610556305) q[19];
rz(0.0616265641447274) q[19];
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
ry(2.4653407644629315) q[0];
rz(2.464869399825181) q[0];
ry(1.462372724702269) q[1];
rz(2.379487502835477) q[1];
ry(-2.3778555828497447) q[2];
rz(2.937936038449495) q[2];
ry(2.852521683842621) q[3];
rz(-0.06393234099051368) q[3];
ry(-1.637577171733684) q[4];
rz(2.663951249228971) q[4];
ry(2.726707907974709) q[5];
rz(2.617443961481165) q[5];
ry(-1.586370067384827) q[6];
rz(-1.1742950001459953) q[6];
ry(2.5474077861536326) q[7];
rz(-1.0612611971750427) q[7];
ry(-1.3675139999771513) q[8];
rz(-2.6682526444464343) q[8];
ry(-3.104001057380328) q[9];
rz(-0.6414411457900222) q[9];
ry(-2.9890659308134326) q[10];
rz(-0.11013152550074069) q[10];
ry(3.049066162322194) q[11];
rz(2.511074382291538) q[11];
ry(-3.0794312445191636) q[12];
rz(-2.0677149533912913) q[12];
ry(1.6830542707919687) q[13];
rz(-1.57208554317313) q[13];
ry(-0.25597172203318685) q[14];
rz(-0.60140339000621) q[14];
ry(1.9613520149111716) q[15];
rz(-1.6263831819522707) q[15];
ry(1.242548925763992) q[16];
rz(2.241182143380975) q[16];
ry(-0.7657602031108324) q[17];
rz(-1.2921167297435598) q[17];
ry(-0.7484824466029982) q[18];
rz(2.776761954433951) q[18];
ry(-1.478491075635325) q[19];
rz(-0.9710960996830856) q[19];
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
ry(2.2699265436905183) q[0];
rz(1.7208422091769808) q[0];
ry(-2.854439775003226) q[1];
rz(-0.44054572623075466) q[1];
ry(-0.5680551843584167) q[2];
rz(3.103509849358964) q[2];
ry(-0.33815794090145623) q[3];
rz(0.15811291307796385) q[3];
ry(3.0476951128869016) q[4];
rz(-0.7596976748615324) q[4];
ry(1.2258132175135277) q[5];
rz(1.8590378515627386) q[5];
ry(0.18094002033846757) q[6];
rz(-1.1941160470278522) q[6];
ry(0.06275298059778028) q[7];
rz(1.0926430656513668) q[7];
ry(3.128101130285152) q[8];
rz(0.20798372328394077) q[8];
ry(-3.0938142739873458) q[9];
rz(1.526255915816107) q[9];
ry(1.2182288912643466) q[10];
rz(-0.7614424044446658) q[10];
ry(-3.0977462836395144) q[11];
rz(-1.2130625016175642) q[11];
ry(1.30065790317693) q[12];
rz(2.7061828193326583) q[12];
ry(-1.1783384260884056) q[13];
rz(-1.2001600324412827) q[13];
ry(-1.0988976057646385) q[14];
rz(-0.23418245613958916) q[14];
ry(-2.155225729820743) q[15];
rz(-1.92006858796305) q[15];
ry(1.6637275446413484) q[16];
rz(1.9303690884814924) q[16];
ry(-3.109083495388744) q[17];
rz(1.3721894608813652) q[17];
ry(-3.1000791920874473) q[18];
rz(-1.072100550953336) q[18];
ry(1.6711142169270499) q[19];
rz(-1.87598775258123) q[19];
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
ry(2.6380464936903087) q[0];
rz(-1.6433873896484643) q[0];
ry(-2.031036505229303) q[1];
rz(2.547877632756768) q[1];
ry(-2.802926459192001) q[2];
rz(-0.5742554859681936) q[2];
ry(-0.44487148326443654) q[3];
rz(2.012903557782489) q[3];
ry(0.6373428879336496) q[4];
rz(1.1439196536664475) q[4];
ry(0.06013536517833717) q[5];
rz(3.029026535678366) q[5];
ry(0.014559704164724074) q[6];
rz(0.7037329071598853) q[6];
ry(-0.734889896571338) q[7];
rz(2.414047964944083) q[7];
ry(-1.6544931235784237) q[8];
rz(-1.660199022810983) q[8];
ry(0.01769867628746424) q[9];
rz(0.5226332829037661) q[9];
ry(3.062988465531863) q[10];
rz(3.068993773116572) q[10];
ry(2.9911681149559857) q[11];
rz(-2.363948496151885) q[11];
ry(-3.103226755631551) q[12];
rz(1.3085724708227726) q[12];
ry(-0.11921405428517043) q[13];
rz(-1.0482378419031468) q[13];
ry(2.764339575139061) q[14];
rz(2.801298259777287) q[14];
ry(-0.00012177012915870478) q[15];
rz(-1.057181095489546) q[15];
ry(0.021284235249021677) q[16];
rz(1.2430890192304513) q[16];
ry(-0.2819782629412275) q[17];
rz(-0.37412973743716277) q[17];
ry(0.8884613081688242) q[18];
rz(0.6862855232480479) q[18];
ry(-0.11227116381525447) q[19];
rz(-0.5966490340220457) q[19];
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
ry(-0.04233145240492526) q[0];
rz(-1.8157444599278385) q[0];
ry(-1.4124145510998465) q[1];
rz(-0.7830124585447501) q[1];
ry(-0.31366458864142377) q[2];
rz(-1.2414137182196852) q[2];
ry(-0.019409917146878005) q[3];
rz(2.583618360947723) q[3];
ry(1.804869350021292) q[4];
rz(-3.091961770345059) q[4];
ry(-1.9908235489659063) q[5];
rz(-2.6309518936095704) q[5];
ry(-1.9322885504596623) q[6];
rz(-0.6828632721105254) q[6];
ry(-1.1565061719958938) q[7];
rz(1.071789014919779) q[7];
ry(0.04830713644613805) q[8];
rz(-1.4137430143173455) q[8];
ry(-1.1042524825038915) q[9];
rz(1.3434861067472013) q[9];
ry(-2.6819936820859454) q[10];
rz(2.29933935961423) q[10];
ry(-2.135810871945801) q[11];
rz(2.7368280046031175) q[11];
ry(-1.9184542521618673) q[12];
rz(-1.6202705188968853) q[12];
ry(2.694122185527442) q[13];
rz(0.6814727784378771) q[13];
ry(1.090488941144163) q[14];
rz(-2.991813386915796) q[14];
ry(-1.710202194393314) q[15];
rz(-1.4260900129023675) q[15];
ry(-1.7802171808749347) q[16];
rz(-0.10297302114796647) q[16];
ry(-0.09695633895464795) q[17];
rz(-0.5739489692329319) q[17];
ry(-0.10636099101748542) q[18];
rz(1.3159004934739071) q[18];
ry(-1.613737281158027) q[19];
rz(-2.9408962554755904) q[19];
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
ry(1.5851316627049934) q[0];
rz(0.7159735550434093) q[0];
ry(0.8923158792384368) q[1];
rz(-2.269125838661635) q[1];
ry(-0.04430242589010014) q[2];
rz(-0.9970626990462067) q[2];
ry(-3.005667228443851) q[3];
rz(3.0658612211892766) q[3];
ry(2.4424715339106573) q[4];
rz(1.6421380808742767) q[4];
ry(-1.109371760601003) q[5];
rz(-1.586674278444045) q[5];
ry(-3.1172992533394734) q[6];
rz(-0.6455860244267412) q[6];
ry(0.2031009158530111) q[7];
rz(0.8349789067063077) q[7];
ry(0.13447036408749832) q[8];
rz(0.2402927174963052) q[8];
ry(0.5228125192602183) q[9];
rz(-0.28460723061823234) q[9];
ry(0.08810931225605935) q[10];
rz(-2.0505487928112114) q[10];
ry(-2.9035000939069295) q[11];
rz(-1.2074363107278678) q[11];
ry(-3.120054900528957) q[12];
rz(-0.3410248863241877) q[12];
ry(-0.13135971556869921) q[13];
rz(1.5385888419757812) q[13];
ry(0.3363927121776049) q[14];
rz(-1.2643393361500668) q[14];
ry(2.998812146109227) q[15];
rz(-0.44775539427005734) q[15];
ry(2.0316297816286433) q[16];
rz(1.0492457481185093) q[16];
ry(-1.7367495091167988) q[17];
rz(2.517116654013474) q[17];
ry(1.11849224464562) q[18];
rz(-3.129772328760655) q[18];
ry(2.9543956776062297) q[19];
rz(0.28114927483319097) q[19];
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
ry(2.5403057446342863) q[0];
rz(-2.0908776769583515) q[0];
ry(0.9979621604216857) q[1];
rz(-1.558719022103512) q[1];
ry(-1.3124618631345453) q[2];
rz(0.7867723020015003) q[2];
ry(1.5377578740827562) q[3];
rz(0.9729823425274534) q[3];
ry(0.9309323773175189) q[4];
rz(-1.9212486681843781) q[4];
ry(0.00286673478086846) q[5];
rz(-1.4575330831477984) q[5];
ry(-2.8695079346373555) q[6];
rz(-1.2683549864235353) q[6];
ry(-2.086695456188493) q[7];
rz(-0.021590709051735137) q[7];
ry(-0.008088505847064553) q[8];
rz(-2.003968529848598) q[8];
ry(0.0841503326856694) q[9];
rz(-3.046058263553287) q[9];
ry(-0.034887204682296336) q[10];
rz(0.29378756018121516) q[10];
ry(0.30132644141495035) q[11];
rz(-2.2655748245697467) q[11];
ry(2.508425991242317) q[12];
rz(-2.523120049168515) q[12];
ry(0.9855296721733559) q[13];
rz(-1.3501970151816851) q[13];
ry(0.1164166870533828) q[14];
rz(-1.0729807586910465) q[14];
ry(1.6774252439068422) q[15];
rz(-1.0807326532916879) q[15];
ry(2.7369527463082277) q[16];
rz(-0.23847412887567046) q[16];
ry(1.4781065048159032) q[17];
rz(-2.9677044320827695) q[17];
ry(-1.3475068989869188) q[18];
rz(1.3653920222177753) q[18];
ry(2.1176721144057673) q[19];
rz(1.603893400058461) q[19];
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
ry(-2.18071499239103) q[0];
rz(0.36659350887122244) q[0];
ry(-0.031067927908540405) q[1];
rz(-1.8902231172855064) q[1];
ry(0.3616550224190602) q[2];
rz(-0.5534156830083699) q[2];
ry(3.1323793531359247) q[3];
rz(-0.4096242879162651) q[3];
ry(0.03241694809491025) q[4];
rz(-1.3311125360698632) q[4];
ry(-1.424904832457064) q[5];
rz(1.7967748161136194) q[5];
ry(0.012162276978532173) q[6];
rz(0.2605168928944206) q[6];
ry(0.20093222005708253) q[7];
rz(-1.4711723619816353) q[7];
ry(0.22938043570589745) q[8];
rz(2.8998447719177673) q[8];
ry(0.5376714494542556) q[9];
rz(0.16394348011691748) q[9];
ry(-3.0387789832587573) q[10];
rz(0.8022759227701197) q[10];
ry(3.117272885916526) q[11];
rz(-0.32109058031416193) q[11];
ry(3.107007588357771) q[12];
rz(-3.1008710184675405) q[12];
ry(3.0659005923611398) q[13];
rz(-1.2437717387160485) q[13];
ry(2.0999620796267218) q[14];
rz(-2.8488176141702755) q[14];
ry(1.5836247728660888) q[15];
rz(-1.5971194801561637) q[15];
ry(3.1044392784503105) q[16];
rz(-3.067961393231669) q[16];
ry(0.623007398054142) q[17];
rz(0.3651230675206047) q[17];
ry(-3.09969683641222) q[18];
rz(-1.7523135132085277) q[18];
ry(1.1806837841581836) q[19];
rz(-2.6580523826327487) q[19];
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
ry(1.4437848857584863) q[0];
rz(-2.3389246670281647) q[0];
ry(2.88468655043066) q[1];
rz(0.028853710771875908) q[1];
ry(1.6771613895211634) q[2];
rz(1.3918418693559806) q[2];
ry(-1.7067533231134402) q[3];
rz(-1.4747637028703795) q[3];
ry(2.221740883114064) q[4];
rz(-2.9299553952556487) q[4];
ry(-0.12877706870587316) q[5];
rz(1.3189300437505436) q[5];
ry(2.8711054959137425) q[6];
rz(0.4855358751317267) q[6];
ry(2.669665327233023) q[7];
rz(-1.1878820468831357) q[7];
ry(1.0835908314036473) q[8];
rz(-1.317657142572223) q[8];
ry(1.273141838365581) q[9];
rz(-1.635457386015056) q[9];
ry(-1.807699487061587) q[10];
rz(1.175714020455162) q[10];
ry(-1.0437474629254826) q[11];
rz(-3.010916822516055) q[11];
ry(-1.6085306131518082) q[12];
rz(-2.327308976484314) q[12];
ry(-2.9294995117124145) q[13];
rz(0.02685708100347501) q[13];
ry(0.07342915751555945) q[14];
rz(-0.27896326432867236) q[14];
ry(3.10555881750183) q[15];
rz(-1.5200986296972483) q[15];
ry(3.1308900869688783) q[16];
rz(-1.973787008056452) q[16];
ry(-0.0653232896405731) q[17];
rz(2.8556563454886073) q[17];
ry(-1.6357173013667021) q[18];
rz(-2.988205932471068) q[18];
ry(-0.3990495861855779) q[19];
rz(0.2666832283833115) q[19];
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
ry(1.3300769696009536) q[0];
rz(0.36257336928602873) q[0];
ry(1.5188994064290533) q[1];
rz(2.3255780558215977) q[1];
ry(-0.1766793691032742) q[2];
rz(-2.8634592868977915) q[2];
ry(0.09340045007750725) q[3];
rz(-0.05236134774859081) q[3];
ry(0.05602857690412624) q[4];
rz(1.4878231331181402) q[4];
ry(-2.0420236899787345) q[5];
rz(-1.714686870194262) q[5];
ry(3.1403048232778192) q[6];
rz(2.017789345638762) q[6];
ry(0.14053405937348984) q[7];
rz(3.0527610440863255) q[7];
ry(3.139444840265517) q[8];
rz(-2.8059598259953775) q[8];
ry(-0.04725522028846285) q[9];
rz(1.955909488790625) q[9];
ry(3.0678158040365346) q[10];
rz(-1.3720189056554393) q[10];
ry(-3.1195878126983656) q[11];
rz(1.4289700470303417) q[11];
ry(-3.0612236806461923) q[12];
rz(2.2369743309103676) q[12];
ry(-3.070218463648993) q[13];
rz(1.229187875359055) q[13];
ry(-1.059551491265701) q[14];
rz(2.1535929863724435) q[14];
ry(1.6135972971734365) q[15];
rz(-2.765346665727068) q[15];
ry(-0.13404652487000135) q[16];
rz(-0.15412155750861078) q[16];
ry(1.369559219116237) q[17];
rz(1.5163793501606067) q[17];
ry(-1.5956880546727599) q[18];
rz(-1.6061434194405786) q[18];
ry(1.7993449333647797) q[19];
rz(-2.905163592462418) q[19];
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
ry(3.1293341345476993) q[0];
rz(-2.045500027631954) q[0];
ry(-0.014425740062685044) q[1];
rz(1.5102524829357584) q[1];
ry(-0.0016574310573425066) q[2];
rz(0.49225940306909205) q[2];
ry(1.3777005036877998) q[3];
rz(0.5826227296816184) q[3];
ry(-1.4940982734865205) q[4];
rz(-2.3842903605806454) q[4];
ry(-1.4933804818975513) q[5];
rz(-0.8367019718967709) q[5];
ry(0.05183079363301994) q[6];
rz(-2.898936484858878) q[6];
ry(1.7402330970745714) q[7];
rz(1.5367111029665494) q[7];
ry(-1.6735211002807306) q[8];
rz(-1.1086372697382765) q[8];
ry(-2.8455252331648975) q[9];
rz(-2.8293867664480516) q[9];
ry(0.007412641438886784) q[10];
rz(-1.8234383535019019) q[10];
ry(1.5192333240668467) q[11];
rz(0.1381757237687389) q[11];
ry(-2.994905070479998) q[12];
rz(-0.553032648087905) q[12];
ry(-0.28520555859191005) q[13];
rz(1.4309555754089054) q[13];
ry(-1.8566605964417096) q[14];
rz(2.4149702652381384) q[14];
ry(-2.2261211119834545) q[15];
rz(1.225847948373067) q[15];
ry(-2.971027200028194) q[16];
rz(1.590453087813124) q[16];
ry(-1.6138866848894722) q[17];
rz(-1.048688332667024) q[17];
ry(-1.565757572889038) q[18];
rz(2.214152797068703) q[18];
ry(-3.131923240894669) q[19];
rz(0.6826217421280524) q[19];