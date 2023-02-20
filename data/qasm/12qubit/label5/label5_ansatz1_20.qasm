OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.8282544622040537) q[0];
rz(-1.4727299521627228) q[0];
ry(-1.8850228995644605) q[1];
rz(-1.5961414482079723) q[1];
ry(1.565078485188324) q[2];
rz(0.9772851439666099) q[2];
ry(1.5703731368136074) q[3];
rz(-0.661415299294613) q[3];
ry(8.30720045357225e-05) q[4];
rz(0.5636046235360284) q[4];
ry(-1.7336542536881352) q[5];
rz(0.6104738171751859) q[5];
ry(-1.9803073612283173) q[6];
rz(1.2919239901869783) q[6];
ry(-0.8125412423163407) q[7];
rz(-1.932370128479044) q[7];
ry(-3.0509290675864715) q[8];
rz(-0.818969063236592) q[8];
ry(-1.9510748211288123) q[9];
rz(-3.127842975177154) q[9];
ry(3.09820415823276) q[10];
rz(2.5525052064979947) q[10];
ry(-1.6500241144170245) q[11];
rz(-2.3572334240954844) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.3890236994347065) q[0];
rz(-1.7896646282772162) q[0];
ry(-1.751059457020444) q[1];
rz(0.5708367283583003) q[1];
ry(-3.140531707519118) q[2];
rz(0.5550859944732478) q[2];
ry(-3.1356036245951886) q[3];
rz(2.213030958791668) q[3];
ry(-1.5716884675175058) q[4];
rz(-0.775444341535228) q[4];
ry(-3.1350082372964443) q[5];
rz(1.309570722314569) q[5];
ry(-3.1394017001457004) q[6];
rz(-2.1844973673649655) q[6];
ry(0.8176339526281766) q[7];
rz(2.10073933266394) q[7];
ry(0.4285440090350141) q[8];
rz(-3.052874734163129) q[8];
ry(-1.13151158705372) q[9];
rz(1.4242286521954166) q[9];
ry(1.069807346942648) q[10];
rz(2.7571301144842835) q[10];
ry(-0.33309057068957965) q[11];
rz(1.3296196733612407) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.2905439434016777) q[0];
rz(-2.0378724309973513) q[0];
ry(-1.676007618634575) q[1];
rz(1.8774020018232838) q[1];
ry(3.0165874627203118) q[2];
rz(0.6514305624233557) q[2];
ry(1.2786960821943563) q[3];
rz(-1.1476972544785105) q[3];
ry(-0.3584841548316957) q[4];
rz(-1.7564676640966546) q[4];
ry(2.627440857852304) q[5];
rz(0.7748070427540421) q[5];
ry(-2.648772832000524) q[6];
rz(-0.7350841464219489) q[6];
ry(-1.2888407202036873) q[7];
rz(0.11360682546120084) q[7];
ry(-2.780738035100375) q[8];
rz(3.0947007804121567) q[8];
ry(-2.983658031550334) q[9];
rz(-0.4668238751705026) q[9];
ry(1.2711838951071865) q[10];
rz(-0.10371870357997626) q[10];
ry(0.07196351831950633) q[11];
rz(0.8025975743591836) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.881601733190185) q[0];
rz(-1.7400329442322564) q[0];
ry(-1.2460798230427625) q[1];
rz(0.9188523788662586) q[1];
ry(-0.00748986777801599) q[2];
rz(0.18233959982142878) q[2];
ry(0.0030445667119309828) q[3];
rz(-0.03486015021513592) q[3];
ry(-0.0030169838055821084) q[4];
rz(-1.3790086579816005) q[4];
ry(-1.5727652674627695) q[5];
rz(-3.1412861678861344) q[5];
ry(-3.1412508303219147) q[6];
rz(-0.3641230265944276) q[6];
ry(1.9352169846601825) q[7];
rz(0.0791752965283421) q[7];
ry(1.6749639844425044) q[8];
rz(-1.6456141865656218) q[8];
ry(-0.288051305218831) q[9];
rz(-0.4247316328613487) q[9];
ry(-1.50463134785343) q[10];
rz(-2.1306017172812224) q[10];
ry(2.33547184695537) q[11];
rz(2.356883721326764) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.599510669523836) q[0];
rz(1.9643455629830733) q[0];
ry(-1.2336943365700925) q[1];
rz(-1.5780587761935703) q[1];
ry(-0.6342924651483293) q[2];
rz(3.036641533425725) q[2];
ry(0.02105009440529848) q[3];
rz(2.9287571824064) q[3];
ry(-2.3465104793091904) q[4];
rz(0.013515375052205058) q[4];
ry(-1.1845215166199283) q[5];
rz(-0.03724102421218323) q[5];
ry(1.2120870175450882) q[6];
rz(-3.043651479407725) q[6];
ry(-2.783262330122614) q[7];
rz(-0.7251255351935747) q[7];
ry(1.5915016809363856) q[8];
rz(0.7354509026570838) q[8];
ry(3.137453524461817) q[9];
rz(-2.27918198464027) q[9];
ry(-2.5899104589125845) q[10];
rz(-2.048196197191045) q[10];
ry(0.08449843381791755) q[11];
rz(-0.6779126764448842) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.770897505055741) q[0];
rz(-1.062505791183492) q[0];
ry(-2.6302645251911523) q[1];
rz(-2.6516976308175955) q[1];
ry(3.1111659627489545) q[2];
rz(-0.1604068938831704) q[2];
ry(0.7399250950887425) q[3];
rz(-0.004812273196021622) q[3];
ry(2.774242702595334) q[4];
rz(-1.5573213551491962) q[4];
ry(0.08118479702532255) q[5];
rz(-2.1013655069168786) q[5];
ry(-1.5272226734518406) q[6];
rz(1.6324097110115678) q[6];
ry(3.1410244889857877) q[7];
rz(2.5776727286144414) q[7];
ry(0.5969555353106237) q[8];
rz(-2.8529301817211503) q[8];
ry(1.576301970213584) q[9];
rz(-1.247461663449696) q[9];
ry(-0.07614829795679195) q[10];
rz(3.057685492297954) q[10];
ry(1.3728121495472818) q[11];
rz(0.6470148740335676) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.31404399384282383) q[0];
rz(-2.33790497404812) q[0];
ry(-1.124838886935382) q[1];
rz(2.6189056233489483) q[1];
ry(0.38363610185903685) q[2];
rz(2.909300534730022) q[2];
ry(1.8750313061374255) q[3];
rz(-2.642393174193763) q[3];
ry(3.1381360375618663) q[4];
rz(1.602704099002513) q[4];
ry(-0.013874197138348237) q[5];
rz(1.388373517316759) q[5];
ry(-1.5280437180016995) q[6];
rz(2.3919038431460797) q[6];
ry(-1.1995904376190383) q[7];
rz(-1.578193028220127) q[7];
ry(-0.011478757961943309) q[8];
rz(-1.2599931723714137) q[8];
ry(3.0726613893355754) q[9];
rz(-2.8729769079363336) q[9];
ry(-1.58996889031382) q[10];
rz(-2.572152791298348) q[10];
ry(-0.7594984458458254) q[11];
rz(0.6351212186561269) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.8453330794232355) q[0];
rz(1.6078525788680644) q[0];
ry(3.1208425413913394) q[1];
rz(-2.771565853810385) q[1];
ry(-0.0014964947697785034) q[2];
rz(2.3798089590673093) q[2];
ry(0.917437831477284) q[3];
rz(2.8157202333737574) q[3];
ry(1.5629130795478927) q[4];
rz(-1.8900557815119754) q[4];
ry(0.8144168827705133) q[5];
rz(-2.8647566030322835) q[5];
ry(0.00033797290644025193) q[6];
rz(-1.1482376765054099) q[6];
ry(-3.1415836185894155) q[7];
rz(-1.5781786186732236) q[7];
ry(3.141462604106612) q[8];
rz(-0.8201751358623931) q[8];
ry(1.5915719294271025) q[9];
rz(-0.3509525670528129) q[9];
ry(2.880859894825522) q[10];
rz(-1.2674234216822944) q[10];
ry(-0.6308080916152686) q[11];
rz(2.236610254027948) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.09255857143024837) q[0];
rz(1.8041742770749232) q[0];
ry(0.5419639909375217) q[1];
rz(2.6772395176625072) q[1];
ry(0.4689125849558217) q[2];
rz(-0.5543199792405361) q[2];
ry(1.5560682782360473) q[3];
rz(0.933008701891363) q[3];
ry(3.0932571514616187) q[4];
rz(2.561521739817853) q[4];
ry(3.140833045486955) q[5];
rz(-0.6465836639124206) q[5];
ry(3.073571903089664) q[6];
rz(3.0989653113904176) q[6];
ry(1.9420990080356475) q[7];
rz(0.7806835937146204) q[7];
ry(1.325467464202811) q[8];
rz(-1.274914678314446) q[8];
ry(-0.008544941434317721) q[9];
rz(2.169719977073372) q[9];
ry(-2.479565490014147) q[10];
rz(-1.5938238254967332) q[10];
ry(1.4417809492536728) q[11];
rz(-2.5721695769280157) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.4175736491677977) q[0];
rz(0.2716035510442767) q[0];
ry(3.128637162557184) q[1];
rz(-1.351874929790597) q[1];
ry(-1.5677982517878866) q[2];
rz(-0.017499922278848615) q[2];
ry(0.016306208774411246) q[3];
rz(-2.4577801360696174) q[3];
ry(1.6036710757258859) q[4];
rz(-1.3238595997318903) q[4];
ry(-1.0359395463234153) q[5];
rz(-2.821479768689349) q[5];
ry(-2.2849089193543803) q[6];
rz(2.409826822957792) q[6];
ry(0.0016463262935100516) q[7];
rz(0.47461403894977144) q[7];
ry(1.4367095097435465) q[8];
rz(2.136121033970782) q[8];
ry(-3.1244759292703415) q[9];
rz(-2.1967853046209918) q[9];
ry(-3.0637248490317464) q[10];
rz(-2.662886860827296) q[10];
ry(0.248316197898081) q[11];
rz(0.5989526852293149) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.348870993329956) q[0];
rz(0.2127053281474787) q[0];
ry(-1.5661681977429192) q[1];
rz(-2.8536242420842015) q[1];
ry(1.9595557049346617) q[2];
rz(-1.578894993563012) q[2];
ry(1.842163125084511) q[3];
rz(-1.4902424350791135) q[3];
ry(3.0590547040083824) q[4];
rz(0.3070439919983679) q[4];
ry(0.0014135646243467193) q[5];
rz(-0.04882816681848403) q[5];
ry(0.0763093009032243) q[6];
rz(2.9115524664948844) q[6];
ry(-1.5703744699436912) q[7];
rz(1.573845960157145) q[7];
ry(-1.2052067776233606) q[8];
rz(1.368622927776322) q[8];
ry(-1.4500255877429178) q[9];
rz(-3.1225430837032504) q[9];
ry(1.8059886340996405) q[10];
rz(-2.2741360547586424) q[10];
ry(1.2720822525785138) q[11];
rz(-1.5066263641432718) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.6436643446006958) q[0];
rz(1.8357721312249842) q[0];
ry(2.979785755779294) q[1];
rz(1.1697346046671502) q[1];
ry(-1.564632650307751) q[2];
rz(-3.019068108002272) q[2];
ry(3.1377775992602954) q[3];
rz(1.6506948196945617) q[3];
ry(-1.4848817282901212) q[4];
rz(1.3990080613209044) q[4];
ry(0.02724977595011363) q[5];
rz(-2.354335215443047) q[5];
ry(-0.0959711061110378) q[6];
rz(-2.276677262423721) q[6];
ry(1.572090799966288) q[7];
rz(2.686314064883685) q[7];
ry(-1.5704444845086944) q[8];
rz(3.14140347566539) q[8];
ry(-0.12969110482046808) q[9];
rz(1.4781704908534592) q[9];
ry(3.0828438335676234) q[10];
rz(0.49883197975294236) q[10];
ry(2.938948447015213) q[11];
rz(1.01360219124804) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.08669008125169243) q[0];
rz(-1.6937311373757842) q[0];
ry(-3.138967118760072) q[1];
rz(1.18699353525513) q[1];
ry(3.1382654148734392) q[2];
rz(0.12501028304897233) q[2];
ry(-1.541334380732458) q[3];
rz(0.7171795881460962) q[3];
ry(1.498951809310051) q[4];
rz(1.6429979005772712) q[4];
ry(3.141379433054269) q[5];
rz(-2.854673979752127) q[5];
ry(-1.4016771301765374) q[6];
rz(-0.03784341903614829) q[6];
ry(-3.141411939075526) q[7];
rz(2.6517773874541573) q[7];
ry(1.5537921361287257) q[8];
rz(-3.118519670704685) q[8];
ry(-1.570716912082856) q[9];
rz(1.5649163036573794) q[9];
ry(-1.7368070772625224) q[10];
rz(-1.4368805484571627) q[10];
ry(2.442619974633558) q[11];
rz(1.82388368333838) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.1383037742298334) q[0];
rz(1.8265251758154628) q[0];
ry(2.970314173688471) q[1];
rz(-1.2530742571992783) q[1];
ry(1.3224810755184553) q[2];
rz(-0.9967937478769775) q[2];
ry(-1.555544329161866) q[3];
rz(1.5586867109408828) q[3];
ry(1.5584816502769372) q[4];
rz(1.5158567839239097) q[4];
ry(4.5380285131457754e-05) q[5];
rz(-1.9834215497146939) q[5];
ry(-3.063761955470669) q[6];
rz(-0.03802785930509659) q[6];
ry(0.00041859962938239903) q[7];
rz(-1.8540956202795853) q[7];
ry(-1.7408632876301464) q[8];
rz(-3.1412258270091478) q[8];
ry(-1.5707090501968441) q[9];
rz(1.5744638489487965) q[9];
ry(-3.1402909527955303) q[10];
rz(0.05913200984674649) q[10];
ry(1.706807527597646) q[11];
rz(-1.8309288340859646) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.9157355706910284) q[0];
rz(2.106518067453158) q[0];
ry(1.5681094968676268) q[1];
rz(-2.966591413765784) q[1];
ry(3.484793985494708e-05) q[2];
rz(-2.130896396510274) q[2];
ry(1.5684607538806414) q[3];
rz(-3.1056275000617393) q[3];
ry(1.3842727071893821) q[4];
rz(0.7914314834914846) q[4];
ry(-0.0015957820684382338) q[5];
rz(1.5486846069654407) q[5];
ry(1.3985348247502198) q[6];
rz(0.0667009634973223) q[6];
ry(3.1411593420313517) q[7];
rz(-2.6020870330110624) q[7];
ry(-2.8656738544504616) q[8];
rz(0.4465257735063508) q[8];
ry(-1.6002060097632764) q[9];
rz(-0.03553892768358047) q[9];
ry(-1.566691804633546) q[10];
rz(-1.5708288658125769) q[10];
ry(-2.0858010223822734) q[11];
rz(-2.2406468735145877) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.5755690346111457) q[0];
rz(0.7546049669310184) q[0];
ry(-0.05689103340453805) q[1];
rz(-3.0101012967046463) q[1];
ry(2.956038164765074) q[2];
rz(-3.0679283869193816) q[2];
ry(1.4285142870100698) q[3];
rz(-0.04170028859790342) q[3];
ry(-0.0991359388687556) q[4];
rz(-0.9065323905628262) q[4];
ry(-3.140051764271409) q[5];
rz(-3.108713036957369) q[5];
ry(-1.3429464335036823) q[6];
rz(2.911457209533928) q[6];
ry(0.0003796642344098098) q[7];
rz(0.7312375111591224) q[7];
ry(-3.1358148420677194) q[8];
rz(-2.69887132944515) q[8];
ry(-0.0011050711766387167) q[9];
rz(-2.3651322724009085) q[9];
ry(-3.0630257299340076) q[10];
rz(-0.6029660985935567) q[10];
ry(2.5507058378232097) q[11];
rz(0.5737392822981888) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.0006411518668018556) q[0];
rz(2.333509665772299) q[0];
ry(-2.5904103487531955) q[1];
rz(-1.308551438959448) q[1];
ry(3.1406490676821845) q[2];
rz(-0.5844542459454454) q[2];
ry(0.38662426795840776) q[3];
rz(0.08124029704696763) q[3];
ry(1.9761894244770994) q[4];
rz(-1.859216116894831) q[4];
ry(0.0019707219122465014) q[5];
rz(0.8316254021299538) q[5];
ry(-3.1409156797958806) q[6];
rz(1.9716398564095758) q[6];
ry(-3.1332574704991143) q[7];
rz(1.7740852129019111) q[7];
ry(1.8309452589186703) q[8];
rz(-3.0570833912442876) q[8];
ry(-1.6650861567789077) q[9];
rz(-1.1949775607074589) q[9];
ry(0.003472517788144046) q[10];
rz(2.137087809837814) q[10];
ry(1.5732034744596273) q[11];
rz(2.258661307107389) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.079263304675713) q[0];
rz(-2.067344912293654) q[0];
ry(1.5781045655455817) q[1];
rz(3.089630181606808) q[1];
ry(-0.0117802880093504) q[2];
rz(2.227148076762525) q[2];
ry(2.5850566580987637) q[3];
rz(0.011148990651772661) q[3];
ry(-0.2944206054113696) q[4];
rz(0.9245783141824546) q[4];
ry(-3.135458059762387) q[5];
rz(-0.3453656057778822) q[5];
ry(2.951416669955314) q[6];
rz(2.2454969160598903) q[6];
ry(2.0108927328908015) q[7];
rz(-1.1093015284988057) q[7];
ry(3.1335524149545346) q[8];
rz(1.4200880581968303) q[8];
ry(3.1411360532989283) q[9];
rz(0.4739946366956497) q[9];
ry(3.1408073606433096) q[10];
rz(3.102963071685829) q[10];
ry(-1.9122706750354297) q[11];
rz(1.0010573916105052) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.000720724799855077) q[0];
rz(-2.617639623503382) q[0];
ry(0.7615922119927391) q[1];
rz(0.08058238319248634) q[1];
ry(1.514653326094824) q[2];
rz(0.019226318713901295) q[2];
ry(-1.9519476559150462) q[3];
rz(2.9159108798117352) q[3];
ry(2.3654866124907854) q[4];
rz(-0.2540211310132699) q[4];
ry(0.0007894535127490607) q[5];
rz(-2.331486019420695) q[5];
ry(3.141406502998905) q[6];
rz(2.6787213413162965) q[6];
ry(3.141439886459629) q[7];
rz(-0.9094783393761964) q[7];
ry(3.141304536535059) q[8];
rz(-1.811991480539031) q[8];
ry(2.548600370295138) q[9];
rz(0.04200538688476297) q[9];
ry(2.033510611784104) q[10];
rz(1.8956988326827542) q[10];
ry(1.8819012424785342) q[11];
rz(-1.2228974625781073) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.6037034781113797) q[0];
rz(1.1402419914109974) q[0];
ry(-0.0129283205845514) q[1];
rz(3.0473885011722013) q[1];
ry(0.07759335624375385) q[2];
rz(-0.24742535620261297) q[2];
ry(3.1414683578913696) q[3];
rz(-1.7640922516882585) q[3];
ry(-0.15814528564151864) q[4];
rz(-1.521944866032925) q[4];
ry(0.0018081790876879598) q[5];
rz(0.5853740645035023) q[5];
ry(2.91680160651007) q[6];
rz(0.1142089690850756) q[6];
ry(1.1730114301490064) q[7];
rz(-1.8380977214309153) q[7];
ry(1.8876088626817729) q[8];
rz(-0.9994745463563123) q[8];
ry(0.0375586041015597) q[9];
rz(1.4843406031608506) q[9];
ry(-0.00017529987706079486) q[10];
rz(-1.5143657393471277) q[10];
ry(2.3938517259879264) q[11];
rz(0.022480705381384757) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.6120081265104194) q[0];
rz(0.6406486427298725) q[0];
ry(-1.60406306450081) q[1];
rz(0.012030377369977469) q[1];
ry(-0.05727418139739907) q[2];
rz(1.545688265500612) q[2];
ry(1.603674587233268) q[3];
rz(0.6477827060262134) q[3];
ry(1.244567341012415) q[4];
rz(0.8386658114115574) q[4];
ry(0.0017907267328223769) q[5];
rz(-0.11953725750637229) q[5];
ry(3.138344466755506) q[6];
rz(-0.27133379590589374) q[6];
ry(-0.7853731832708943) q[7];
rz(-2.499312007891107) q[7];
ry(-3.1130500142657747) q[8];
rz(-0.9979799746090842) q[8];
ry(0.006550259783222323) q[9];
rz(-1.392551198547335) q[9];
ry(-0.0034059924060780844) q[10];
rz(-0.3825955705250923) q[10];
ry(-1.7318325401372354) q[11];
rz(1.534930189132342) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.03347863731072831) q[0];
rz(-0.6282865831347768) q[0];
ry(-3.1411440721577475) q[1];
rz(-2.2699650997210714) q[1];
ry(0.9625713651568857) q[2];
rz(-1.1573971105780716) q[2];
ry(4.0307386653205064e-05) q[3];
rz(2.9731940172140723) q[3];
ry(-0.0035718540775228767) q[4];
rz(-1.5143445512455571) q[4];
ry(3.1394951558359123) q[5];
rz(-1.7327104439994931) q[5];
ry(2.463913699982432) q[6];
rz(0.4398244263366958) q[6];
ry(0.03293634102579191) q[7];
rz(0.20384451497635414) q[7];
ry(-1.5781807382615465) q[8];
rz(2.5833051674406167) q[8];
ry(1.5887175379208023) q[9];
rz(-3.137075693365725) q[9];
ry(1.5715954875629654) q[10];
rz(0.6503565103075121) q[10];
ry(-1.5298749476555602) q[11];
rz(-0.8253848086962283) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.527730954064708) q[0];
rz(0.5244956715264578) q[0];
ry(3.141010969445309) q[1];
rz(2.418058467077848) q[1];
ry(-3.1411112599860638) q[2];
rz(-2.874710517449531) q[2];
ry(-3.1412820023829644) q[3];
rz(-0.8417918046076815) q[3];
ry(2.148442352660364) q[4];
rz(-1.302708372475073) q[4];
ry(-0.0009369392550898056) q[5];
rz(1.2067419216850757) q[5];
ry(-3.1405488304078446) q[6];
rz(2.7252948581727274) q[6];
ry(-0.00035594691198514994) q[7];
rz(0.7249202787374411) q[7];
ry(-0.0005470019079155125) q[8];
rz(-1.012169439520349) q[8];
ry(1.570872247457058) q[9];
rz(1.5710115539521956) q[9];
ry(-0.02206323965065149) q[10];
rz(0.9206562628716201) q[10];
ry(-1.5723049587535716) q[11];
rz(-2.0442432292264714) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.8950205525026139) q[0];
rz(2.7934566325878945) q[0];
ry(-1.5683481098299934) q[1];
rz(2.1091564780846834) q[1];
ry(1.7777162181459365) q[2];
rz(1.9318591203805147) q[2];
ry(-3.141055088028786) q[3];
rz(2.393832027969247) q[3];
ry(-1.5845226982513851) q[4];
rz(0.9838158381892009) q[4];
ry(1.571411766053919) q[5];
rz(-2.5837599588481908) q[5];
ry(1.0855299726807603) q[6];
rz(1.1717552978419623) q[6];
ry(1.5905439495369507) q[7];
rz(-2.3661744636143056) q[7];
ry(1.5708302970493646) q[8];
rz(-2.462533680875293) q[8];
ry(-1.5708296156526016) q[9];
rz(0.5786135325174355) q[9];
ry(-1.5707070897915063) q[10];
rz(-0.8907160594148645) q[10];
ry(-3.1404192857339197) q[11];
rz(0.09144864038925468) q[11];