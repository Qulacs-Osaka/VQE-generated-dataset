OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.0897249810744327) q[0];
ry(1.3475733091844306) q[1];
cx q[0],q[1];
ry(-2.684832516185088) q[0];
ry(2.028220867327331) q[1];
cx q[0],q[1];
ry(-0.16270660474035645) q[1];
ry(-2.0331749885257127) q[2];
cx q[1],q[2];
ry(0.04776271702471124) q[1];
ry(-0.43541019296061023) q[2];
cx q[1],q[2];
ry(-2.921161637580837) q[2];
ry(1.9959759600303326) q[3];
cx q[2],q[3];
ry(-0.7472655545924207) q[2];
ry(-2.376475338402838) q[3];
cx q[2],q[3];
ry(3.0309876352457583) q[3];
ry(-2.549134509447914) q[4];
cx q[3],q[4];
ry(-0.29137609547712895) q[3];
ry(1.3937773261398645) q[4];
cx q[3],q[4];
ry(-1.5402707434937644) q[4];
ry(-2.3391326439217686) q[5];
cx q[4],q[5];
ry(-0.44444076033829205) q[4];
ry(-1.290252430796309) q[5];
cx q[4],q[5];
ry(-3.1409883495940165) q[5];
ry(1.8535775715658698) q[6];
cx q[5],q[6];
ry(1.6882246856016854) q[5];
ry(0.6852105259355472) q[6];
cx q[5],q[6];
ry(-1.427733417475779) q[6];
ry(-0.8167639055479151) q[7];
cx q[6],q[7];
ry(-1.3859459335061683) q[6];
ry(-1.0391618416611692) q[7];
cx q[6],q[7];
ry(-0.5962438529968068) q[0];
ry(2.9823114562447945) q[1];
cx q[0],q[1];
ry(1.7137339101979625) q[0];
ry(2.337871433014322) q[1];
cx q[0],q[1];
ry(-2.587947064049994) q[1];
ry(1.3417846025465785) q[2];
cx q[1],q[2];
ry(-2.313951704543715) q[1];
ry(1.3856647757333524) q[2];
cx q[1],q[2];
ry(-1.2945750716647204) q[2];
ry(0.0026071473624973552) q[3];
cx q[2],q[3];
ry(1.0227293565061926) q[2];
ry(1.1920840877870456) q[3];
cx q[2],q[3];
ry(0.45452825990159607) q[3];
ry(-1.3653594999602605) q[4];
cx q[3],q[4];
ry(2.2294102729393526) q[3];
ry(1.159491409639245) q[4];
cx q[3],q[4];
ry(0.913118711022325) q[4];
ry(2.817920377407223) q[5];
cx q[4],q[5];
ry(0.944022326286923) q[4];
ry(-0.9195112626867868) q[5];
cx q[4],q[5];
ry(-1.0975162666621492) q[5];
ry(1.9572228613216864) q[6];
cx q[5],q[6];
ry(-2.9648489126114175) q[5];
ry(2.2982520689286243) q[6];
cx q[5],q[6];
ry(1.693054085855291) q[6];
ry(-1.0329353930992697) q[7];
cx q[6],q[7];
ry(-2.9308111254524585) q[6];
ry(1.6819429791398628) q[7];
cx q[6],q[7];
ry(2.399069789704203) q[0];
ry(2.1610812353757374) q[1];
cx q[0],q[1];
ry(-1.3409958197126357) q[0];
ry(0.5941793673443999) q[1];
cx q[0],q[1];
ry(-2.105598856954457) q[1];
ry(-1.489086506243917) q[2];
cx q[1],q[2];
ry(3.123752153218729) q[1];
ry(-1.800810608371223) q[2];
cx q[1],q[2];
ry(-2.650130383631218) q[2];
ry(-2.026627059637649) q[3];
cx q[2],q[3];
ry(-2.646143260632465) q[2];
ry(-1.2943428429103208) q[3];
cx q[2],q[3];
ry(-1.0949187149412198) q[3];
ry(1.8690771136891096) q[4];
cx q[3],q[4];
ry(-2.8417577936668796) q[3];
ry(-1.896583796830661) q[4];
cx q[3],q[4];
ry(0.1612577514462199) q[4];
ry(-0.5886562581320635) q[5];
cx q[4],q[5];
ry(0.343904987695419) q[4];
ry(-2.5721898522888464) q[5];
cx q[4],q[5];
ry(-1.1982316366926664) q[5];
ry(0.23497208248320894) q[6];
cx q[5],q[6];
ry(-3.058133265214955) q[5];
ry(1.7584850943727444) q[6];
cx q[5],q[6];
ry(1.9619500860434642) q[6];
ry(-0.1380830056475356) q[7];
cx q[6],q[7];
ry(0.2847622181419052) q[6];
ry(2.644628618872842) q[7];
cx q[6],q[7];
ry(0.9800496656211548) q[0];
ry(1.9919958932092179) q[1];
cx q[0],q[1];
ry(1.264308554413022) q[0];
ry(2.1275925595563994) q[1];
cx q[0],q[1];
ry(-0.5498633958080368) q[1];
ry(-0.34092300757474087) q[2];
cx q[1],q[2];
ry(-1.3260633793590806) q[1];
ry(-0.22841210883340768) q[2];
cx q[1],q[2];
ry(-1.1652578162466725) q[2];
ry(-2.757258022942738) q[3];
cx q[2],q[3];
ry(-2.374908235647846) q[2];
ry(-2.3140289419819076) q[3];
cx q[2],q[3];
ry(0.25783609737397195) q[3];
ry(-0.09672959502256477) q[4];
cx q[3],q[4];
ry(-1.6022182969942984) q[3];
ry(0.3382729784439729) q[4];
cx q[3],q[4];
ry(-0.27756608715632713) q[4];
ry(0.6848528655257035) q[5];
cx q[4],q[5];
ry(2.0726817809191598) q[4];
ry(1.2815822688921985) q[5];
cx q[4],q[5];
ry(-1.529404480380516) q[5];
ry(-2.704398202440901) q[6];
cx q[5],q[6];
ry(-2.8207204525441973) q[5];
ry(-0.6328495422051414) q[6];
cx q[5],q[6];
ry(1.1170073379407643) q[6];
ry(-2.0010040501734654) q[7];
cx q[6],q[7];
ry(-1.8727722695820372) q[6];
ry(-1.3863055712464751) q[7];
cx q[6],q[7];
ry(-1.4224929007455192) q[0];
ry(1.4431055348501314) q[1];
cx q[0],q[1];
ry(0.32738472972876664) q[0];
ry(-0.11403048329237772) q[1];
cx q[0],q[1];
ry(0.4899427176327116) q[1];
ry(-1.2051952705989528) q[2];
cx q[1],q[2];
ry(-1.83016690562337) q[1];
ry(0.5998813033874841) q[2];
cx q[1],q[2];
ry(0.7946541990230336) q[2];
ry(1.6480756808047163) q[3];
cx q[2],q[3];
ry(-2.7890024849173076) q[2];
ry(0.8983258500899316) q[3];
cx q[2],q[3];
ry(2.912206911655571) q[3];
ry(-3.0196772765492277) q[4];
cx q[3],q[4];
ry(-0.3388603519789841) q[3];
ry(-2.518279464358469) q[4];
cx q[3],q[4];
ry(2.5675982885379076) q[4];
ry(-1.4250784734765056) q[5];
cx q[4],q[5];
ry(3.069900192231902) q[4];
ry(3.130116124055464) q[5];
cx q[4],q[5];
ry(1.8271035687670354) q[5];
ry(-2.863182594889585) q[6];
cx q[5],q[6];
ry(-2.437520717214784) q[5];
ry(1.6544954670093437) q[6];
cx q[5],q[6];
ry(-1.981042490945618) q[6];
ry(-2.6379449887851436) q[7];
cx q[6],q[7];
ry(1.5456658464685518) q[6];
ry(2.120839282760366) q[7];
cx q[6],q[7];
ry(1.9167229126136842) q[0];
ry(-0.9078777697836582) q[1];
cx q[0],q[1];
ry(-2.3985723294691335) q[0];
ry(0.34226035201579563) q[1];
cx q[0],q[1];
ry(-0.9005182860923986) q[1];
ry(2.0270250837627346) q[2];
cx q[1],q[2];
ry(-2.895917147162402) q[1];
ry(-0.6916439644118283) q[2];
cx q[1],q[2];
ry(1.541159836074329) q[2];
ry(-1.8504013784634874) q[3];
cx q[2],q[3];
ry(1.46096906650947) q[2];
ry(2.773925639066281) q[3];
cx q[2],q[3];
ry(1.064071224202162) q[3];
ry(-2.658934911336114) q[4];
cx q[3],q[4];
ry(-2.3246971708308637) q[3];
ry(1.0523104060002482) q[4];
cx q[3],q[4];
ry(-1.4270926009768123) q[4];
ry(0.4048640177987215) q[5];
cx q[4],q[5];
ry(0.34564333365240074) q[4];
ry(-1.5843380553629776) q[5];
cx q[4],q[5];
ry(-0.9392269652935432) q[5];
ry(-3.0307910545381356) q[6];
cx q[5],q[6];
ry(2.3060858904758894) q[5];
ry(-0.2918641671658131) q[6];
cx q[5],q[6];
ry(3.1208740816547045) q[6];
ry(-0.1946471936999501) q[7];
cx q[6],q[7];
ry(-1.1200050072889447) q[6];
ry(0.20560008538857877) q[7];
cx q[6],q[7];
ry(0.573818193906211) q[0];
ry(-2.3210312224452343) q[1];
cx q[0],q[1];
ry(-1.396016646041561) q[0];
ry(2.8501694818811205) q[1];
cx q[0],q[1];
ry(2.0557953994831024) q[1];
ry(2.7057925062101256) q[2];
cx q[1],q[2];
ry(0.8807745082879519) q[1];
ry(-1.2521683245815132) q[2];
cx q[1],q[2];
ry(-0.7904149803315881) q[2];
ry(1.8799614501158715) q[3];
cx q[2],q[3];
ry(2.5270176648404585) q[2];
ry(0.28138531185449533) q[3];
cx q[2],q[3];
ry(-1.8414566381731472) q[3];
ry(2.345492441196962) q[4];
cx q[3],q[4];
ry(3.080836396242681) q[3];
ry(-0.2708039433084412) q[4];
cx q[3],q[4];
ry(2.3779392360145084) q[4];
ry(-0.1999796733393154) q[5];
cx q[4],q[5];
ry(0.2849852876912804) q[4];
ry(0.8488613037618551) q[5];
cx q[4],q[5];
ry(-0.30785104231867827) q[5];
ry(-2.4652628864453483) q[6];
cx q[5],q[6];
ry(0.2334800402616376) q[5];
ry(1.536485753553019) q[6];
cx q[5],q[6];
ry(1.6196981073456966) q[6];
ry(-0.35880584101025936) q[7];
cx q[6],q[7];
ry(-3.0004056565621835) q[6];
ry(1.5516577652926866) q[7];
cx q[6],q[7];
ry(2.8266783108706313) q[0];
ry(-1.2719244221302088) q[1];
cx q[0],q[1];
ry(1.0944598132212628) q[0];
ry(-0.444053790454495) q[1];
cx q[0],q[1];
ry(-1.4400633698815024) q[1];
ry(1.646914777523846) q[2];
cx q[1],q[2];
ry(2.497938885435624) q[1];
ry(-2.645874348658705) q[2];
cx q[1],q[2];
ry(-3.00386566348784) q[2];
ry(-2.661905577791654) q[3];
cx q[2],q[3];
ry(-1.4380095114785698) q[2];
ry(0.7435177943391196) q[3];
cx q[2],q[3];
ry(1.7984342982371713) q[3];
ry(0.46141496223816425) q[4];
cx q[3],q[4];
ry(1.4624773659251522) q[3];
ry(-1.7860902673338468) q[4];
cx q[3],q[4];
ry(0.2633632635066708) q[4];
ry(-0.19970378136066902) q[5];
cx q[4],q[5];
ry(2.5166556305155234) q[4];
ry(3.1244833061328756) q[5];
cx q[4],q[5];
ry(0.8744804506563334) q[5];
ry(0.5125321668520941) q[6];
cx q[5],q[6];
ry(-0.18260717626690326) q[5];
ry(1.0985552188566325) q[6];
cx q[5],q[6];
ry(-0.3385326754921225) q[6];
ry(-2.5253979836072498) q[7];
cx q[6],q[7];
ry(-1.658158819980393) q[6];
ry(2.246713574593455) q[7];
cx q[6],q[7];
ry(0.6804592909660573) q[0];
ry(0.13166629102284422) q[1];
cx q[0],q[1];
ry(-0.48595092406541623) q[0];
ry(2.40469534179785) q[1];
cx q[0],q[1];
ry(-0.43830233351193965) q[1];
ry(0.4662958791873546) q[2];
cx q[1],q[2];
ry(-0.6889089104309134) q[1];
ry(-0.015745354208346107) q[2];
cx q[1],q[2];
ry(2.303271525887075) q[2];
ry(-1.9483273139945552) q[3];
cx q[2],q[3];
ry(-0.37964402206498704) q[2];
ry(1.8641439616892264) q[3];
cx q[2],q[3];
ry(0.30135570568536885) q[3];
ry(2.254605076130992) q[4];
cx q[3],q[4];
ry(-1.493708847345717) q[3];
ry(-1.6646351147070084) q[4];
cx q[3],q[4];
ry(-1.1762765057130682) q[4];
ry(-2.0458305278602404) q[5];
cx q[4],q[5];
ry(-2.994739325227435) q[4];
ry(-0.9844259989684553) q[5];
cx q[4],q[5];
ry(2.672965972102897) q[5];
ry(1.2209003283330162) q[6];
cx q[5],q[6];
ry(-2.7314925286490217) q[5];
ry(2.810423275250125) q[6];
cx q[5],q[6];
ry(-3.093194779491716) q[6];
ry(0.11009245520252253) q[7];
cx q[6],q[7];
ry(-2.1546897632427475) q[6];
ry(-2.181738395269659) q[7];
cx q[6],q[7];
ry(1.954498668036801) q[0];
ry(1.4757514735811) q[1];
cx q[0],q[1];
ry(-2.5979990169518614) q[0];
ry(-2.6536862098134377) q[1];
cx q[0],q[1];
ry(-0.2517249026056225) q[1];
ry(0.8199010463610366) q[2];
cx q[1],q[2];
ry(-2.2176071625099167) q[1];
ry(0.08662328924591911) q[2];
cx q[1],q[2];
ry(0.229583403781483) q[2];
ry(1.8107251384072383) q[3];
cx q[2],q[3];
ry(0.3894114597993692) q[2];
ry(1.696121073670304) q[3];
cx q[2],q[3];
ry(-2.369451566555776) q[3];
ry(3.1250807534782212) q[4];
cx q[3],q[4];
ry(2.3933741788878073) q[3];
ry(1.3227257324673412) q[4];
cx q[3],q[4];
ry(2.6828569275541017) q[4];
ry(2.2373416772905523) q[5];
cx q[4],q[5];
ry(-2.217126871733211) q[4];
ry(-0.9589207192502102) q[5];
cx q[4],q[5];
ry(-2.708651009165798) q[5];
ry(-0.3214793616827043) q[6];
cx q[5],q[6];
ry(-1.9514963278539543) q[5];
ry(-3.039687609872798) q[6];
cx q[5],q[6];
ry(-2.794843320353244) q[6];
ry(2.372575284887622) q[7];
cx q[6],q[7];
ry(-0.642928429529043) q[6];
ry(0.3652044969317325) q[7];
cx q[6],q[7];
ry(0.8185473057186184) q[0];
ry(-2.74947073244005) q[1];
cx q[0],q[1];
ry(-0.09940549571101999) q[0];
ry(0.8922107988969862) q[1];
cx q[0],q[1];
ry(1.0758957404310634) q[1];
ry(-1.2661107279809898) q[2];
cx q[1],q[2];
ry(-0.33432557375184263) q[1];
ry(-1.915530577491876) q[2];
cx q[1],q[2];
ry(-1.4340337414486308) q[2];
ry(2.2996959482884285) q[3];
cx q[2],q[3];
ry(0.03446587849446332) q[2];
ry(-0.12252609200417465) q[3];
cx q[2],q[3];
ry(0.42189669908495286) q[3];
ry(2.109711496946309) q[4];
cx q[3],q[4];
ry(-2.0486423199978407) q[3];
ry(1.9245196022902862) q[4];
cx q[3],q[4];
ry(0.027545397053708953) q[4];
ry(1.3262764206741346) q[5];
cx q[4],q[5];
ry(-2.1883858281511257) q[4];
ry(2.3694323491632496) q[5];
cx q[4],q[5];
ry(0.8486341978285799) q[5];
ry(1.8750175750609959) q[6];
cx q[5],q[6];
ry(-1.6146347968877808) q[5];
ry(0.290753983477766) q[6];
cx q[5],q[6];
ry(-2.773776440678872) q[6];
ry(1.6043795094809472) q[7];
cx q[6],q[7];
ry(-1.5300363607829761) q[6];
ry(2.811837847211566) q[7];
cx q[6],q[7];
ry(-0.5391989703880299) q[0];
ry(-2.4486275281649315) q[1];
cx q[0],q[1];
ry(-3.047868253705848) q[0];
ry(1.5970849175789177) q[1];
cx q[0],q[1];
ry(-1.5480991239932065) q[1];
ry(2.0640995324865337) q[2];
cx q[1],q[2];
ry(1.2896902066777405) q[1];
ry(2.837038126377973) q[2];
cx q[1],q[2];
ry(-2.642517544926302) q[2];
ry(-0.676226855765707) q[3];
cx q[2],q[3];
ry(-2.0805491960360514) q[2];
ry(1.7553824091002) q[3];
cx q[2],q[3];
ry(-1.454243238932287) q[3];
ry(2.2123667339144752) q[4];
cx q[3],q[4];
ry(-1.6388600683091807) q[3];
ry(1.230817168787544) q[4];
cx q[3],q[4];
ry(-2.4034597857250257) q[4];
ry(-1.3010068422664856) q[5];
cx q[4],q[5];
ry(0.5637246434186807) q[4];
ry(1.9432939057762881) q[5];
cx q[4],q[5];
ry(-1.5407418366192855) q[5];
ry(-1.7133441028803287) q[6];
cx q[5],q[6];
ry(-2.0686122886320493) q[5];
ry(1.7934061973991549) q[6];
cx q[5],q[6];
ry(3.081360390543174) q[6];
ry(-0.8544469081468836) q[7];
cx q[6],q[7];
ry(1.749243551637915) q[6];
ry(-0.21935601621608358) q[7];
cx q[6],q[7];
ry(-2.1702760608246057) q[0];
ry(-0.964045251822581) q[1];
cx q[0],q[1];
ry(-0.1137705374680058) q[0];
ry(2.7601361392232593) q[1];
cx q[0],q[1];
ry(0.2866002850470016) q[1];
ry(-0.711973944818129) q[2];
cx q[1],q[2];
ry(1.2063158584597158) q[1];
ry(-0.6536189023517761) q[2];
cx q[1],q[2];
ry(-0.923322546959592) q[2];
ry(-2.175700578040395) q[3];
cx q[2],q[3];
ry(-1.5391557689803104) q[2];
ry(2.7418945298974813) q[3];
cx q[2],q[3];
ry(1.301143123384593) q[3];
ry(-2.307545811270438) q[4];
cx q[3],q[4];
ry(-1.7845564991001743) q[3];
ry(-2.4873792330582285) q[4];
cx q[3],q[4];
ry(2.0806599856950694) q[4];
ry(2.5847314246789077) q[5];
cx q[4],q[5];
ry(2.719335043081734) q[4];
ry(2.961904104309623) q[5];
cx q[4],q[5];
ry(1.4770431757485523) q[5];
ry(-1.509388149577176) q[6];
cx q[5],q[6];
ry(2.8964277549991317) q[5];
ry(-1.5944298957049492) q[6];
cx q[5],q[6];
ry(1.4973680236811542) q[6];
ry(-2.309001541308037) q[7];
cx q[6],q[7];
ry(-1.2834286840168696) q[6];
ry(-2.658781678937924) q[7];
cx q[6],q[7];
ry(-0.07202723190555815) q[0];
ry(-2.4404475715296865) q[1];
cx q[0],q[1];
ry(0.3958692868726973) q[0];
ry(-1.6095383890359125) q[1];
cx q[0],q[1];
ry(0.5733556654922465) q[1];
ry(-0.1906956735086216) q[2];
cx q[1],q[2];
ry(1.1153682213699725) q[1];
ry(2.875074649896612) q[2];
cx q[1],q[2];
ry(1.5663001112951886) q[2];
ry(2.07307364465718) q[3];
cx q[2],q[3];
ry(-1.0405319758252234) q[2];
ry(0.05589915734954914) q[3];
cx q[2],q[3];
ry(1.375477020204994) q[3];
ry(3.1192323029046882) q[4];
cx q[3],q[4];
ry(3.015012796129988) q[3];
ry(1.4918193742565968) q[4];
cx q[3],q[4];
ry(2.665147910742717) q[4];
ry(-2.4247416903740078) q[5];
cx q[4],q[5];
ry(-0.4302867186078432) q[4];
ry(-0.20675480146144024) q[5];
cx q[4],q[5];
ry(-1.9634974217201) q[5];
ry(1.6162943147216509) q[6];
cx q[5],q[6];
ry(-2.046048873485403) q[5];
ry(1.727541137495999) q[6];
cx q[5],q[6];
ry(0.08940224184053491) q[6];
ry(0.6902702236817851) q[7];
cx q[6],q[7];
ry(2.5838023355427997) q[6];
ry(2.6750636346254826) q[7];
cx q[6],q[7];
ry(-1.6105747362204421) q[0];
ry(0.7899248434949195) q[1];
cx q[0],q[1];
ry(2.997569428235065) q[0];
ry(2.8735914721589992) q[1];
cx q[0],q[1];
ry(-2.7411634644743548) q[1];
ry(-2.0226621936995555) q[2];
cx q[1],q[2];
ry(1.850411307325368) q[1];
ry(2.8482255221209787) q[2];
cx q[1],q[2];
ry(-0.4393722508148281) q[2];
ry(-1.7323522801060816) q[3];
cx q[2],q[3];
ry(-1.353618150401525) q[2];
ry(-2.644411242882436) q[3];
cx q[2],q[3];
ry(-2.966591116806409) q[3];
ry(2.6955134184558616) q[4];
cx q[3],q[4];
ry(-0.16683209627991066) q[3];
ry(-3.0195121245381653) q[4];
cx q[3],q[4];
ry(1.3041782674275493) q[4];
ry(-0.779169801007972) q[5];
cx q[4],q[5];
ry(1.5300996724360845) q[4];
ry(1.2756074864522222) q[5];
cx q[4],q[5];
ry(0.7232276809554258) q[5];
ry(0.17281086276447635) q[6];
cx q[5],q[6];
ry(0.4741394345123373) q[5];
ry(1.793904591190037) q[6];
cx q[5],q[6];
ry(0.1876381487588299) q[6];
ry(2.042370015467796) q[7];
cx q[6],q[7];
ry(1.110902767663859) q[6];
ry(0.8932992177955342) q[7];
cx q[6],q[7];
ry(-1.1417694096449693) q[0];
ry(-0.7069352877651426) q[1];
cx q[0],q[1];
ry(-0.44185007454929526) q[0];
ry(-2.0892019100065227) q[1];
cx q[0],q[1];
ry(-2.2154662686245796) q[1];
ry(-2.5906422671298377) q[2];
cx q[1],q[2];
ry(-0.3429503253531055) q[1];
ry(-0.013780252235333634) q[2];
cx q[1],q[2];
ry(1.789426486453542) q[2];
ry(1.5353481369163138) q[3];
cx q[2],q[3];
ry(1.9467476220348592) q[2];
ry(1.984095055609309) q[3];
cx q[2],q[3];
ry(2.3654694524908306) q[3];
ry(-0.031774021190904764) q[4];
cx q[3],q[4];
ry(2.6559983157582705) q[3];
ry(0.826167721790835) q[4];
cx q[3],q[4];
ry(-1.946100050904433) q[4];
ry(3.1231713795942575) q[5];
cx q[4],q[5];
ry(1.2939241530734664) q[4];
ry(-0.5835143155079283) q[5];
cx q[4],q[5];
ry(-1.8948902921093973) q[5];
ry(-1.1743287993126268) q[6];
cx q[5],q[6];
ry(-1.3266329834460955) q[5];
ry(0.4750744379478766) q[6];
cx q[5],q[6];
ry(-0.3799608004392902) q[6];
ry(0.5455944143612346) q[7];
cx q[6],q[7];
ry(2.945012293741821) q[6];
ry(-1.34898539082219) q[7];
cx q[6],q[7];
ry(-1.80694110683375) q[0];
ry(-1.9455500954003107) q[1];
cx q[0],q[1];
ry(1.0453012611326846) q[0];
ry(-2.843133359275958) q[1];
cx q[0],q[1];
ry(2.967441442380809) q[1];
ry(2.953252668113424) q[2];
cx q[1],q[2];
ry(1.8692500055434627) q[1];
ry(2.571755665412428) q[2];
cx q[1],q[2];
ry(1.4040419065803713) q[2];
ry(2.446390430215103) q[3];
cx q[2],q[3];
ry(0.169322809985883) q[2];
ry(1.2562162770379437) q[3];
cx q[2],q[3];
ry(-0.5340617864157844) q[3];
ry(0.5851246011671939) q[4];
cx q[3],q[4];
ry(2.365057324164906) q[3];
ry(-0.8509187589496161) q[4];
cx q[3],q[4];
ry(-2.215275098085912) q[4];
ry(-2.0282519086936124) q[5];
cx q[4],q[5];
ry(3.045261093957773) q[4];
ry(3.1398237538556426) q[5];
cx q[4],q[5];
ry(2.735586844495005) q[5];
ry(0.5164617280604498) q[6];
cx q[5],q[6];
ry(-1.881697753664436) q[5];
ry(-2.0979352690249327) q[6];
cx q[5],q[6];
ry(-2.276666981311803) q[6];
ry(-2.2348471587113066) q[7];
cx q[6],q[7];
ry(-2.316940288138629) q[6];
ry(-1.0234468116403956) q[7];
cx q[6],q[7];
ry(-2.0948394806336847) q[0];
ry(0.21580741373828616) q[1];
cx q[0],q[1];
ry(-2.2690154876823767) q[0];
ry(2.99446735596656) q[1];
cx q[0],q[1];
ry(1.1164277424539186) q[1];
ry(-0.8780269495353893) q[2];
cx q[1],q[2];
ry(2.758725505015814) q[1];
ry(0.9856145056656542) q[2];
cx q[1],q[2];
ry(1.8386536546399073) q[2];
ry(-1.8398434473758405) q[3];
cx q[2],q[3];
ry(2.0099223360142364) q[2];
ry(2.851504507734783) q[3];
cx q[2],q[3];
ry(0.5638181356484819) q[3];
ry(-2.4112486367699546) q[4];
cx q[3],q[4];
ry(-1.391412276716232) q[3];
ry(0.41213755608265146) q[4];
cx q[3],q[4];
ry(-1.7520232394560384) q[4];
ry(-0.6964903493239714) q[5];
cx q[4],q[5];
ry(-1.0375265549264538) q[4];
ry(3.020087707964516) q[5];
cx q[4],q[5];
ry(1.1907877139347742) q[5];
ry(0.4030688597394824) q[6];
cx q[5],q[6];
ry(0.589134530579418) q[5];
ry(-3.090574177543396) q[6];
cx q[5],q[6];
ry(1.9951570203630626) q[6];
ry(3.0076235803756366) q[7];
cx q[6],q[7];
ry(2.455556605835862) q[6];
ry(1.2006223015903696) q[7];
cx q[6],q[7];
ry(0.15282046379774264) q[0];
ry(-1.1271791836533944) q[1];
cx q[0],q[1];
ry(-2.6088194340290554) q[0];
ry(2.3174065420386025) q[1];
cx q[0],q[1];
ry(-2.5761081557208465) q[1];
ry(1.820226329462499) q[2];
cx q[1],q[2];
ry(1.4986789086088406) q[1];
ry(-1.3477588561286176) q[2];
cx q[1],q[2];
ry(0.6324667573434128) q[2];
ry(-2.992638809940971) q[3];
cx q[2],q[3];
ry(-2.5788371274785766) q[2];
ry(-1.1660350438850284) q[3];
cx q[2],q[3];
ry(2.6904011576139752) q[3];
ry(1.5075357654303707) q[4];
cx q[3],q[4];
ry(-1.0651170574799877) q[3];
ry(-2.2937437665014304) q[4];
cx q[3],q[4];
ry(1.4342345726031729) q[4];
ry(2.668675704735598) q[5];
cx q[4],q[5];
ry(-0.6531163717308708) q[4];
ry(2.152883686295946) q[5];
cx q[4],q[5];
ry(-2.480403921327325) q[5];
ry(-2.583115252185939) q[6];
cx q[5],q[6];
ry(-2.977123991850119) q[5];
ry(-1.5655905364474858) q[6];
cx q[5],q[6];
ry(2.929749942034979) q[6];
ry(-1.3802264989330917) q[7];
cx q[6],q[7];
ry(-0.375812170890069) q[6];
ry(2.439193217091508) q[7];
cx q[6],q[7];
ry(2.922914831283309) q[0];
ry(-1.5729956137517558) q[1];
ry(0.09469468912619482) q[2];
ry(2.9263628303377516) q[3];
ry(-1.3422084144580957) q[4];
ry(-0.5847927955209844) q[5];
ry(0.20666080169040235) q[6];
ry(1.9537760374081439) q[7];