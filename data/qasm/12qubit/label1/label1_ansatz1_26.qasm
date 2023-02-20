OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.6255766445797266) q[0];
rz(-1.7182009116954635) q[0];
ry(2.0931968786845516) q[1];
rz(-1.616727139657562) q[1];
ry(-0.003393391016175862) q[2];
rz(-0.21125341591693123) q[2];
ry(0.6558861726555014) q[3];
rz(0.8242719220263979) q[3];
ry(0.45362766003290594) q[4];
rz(2.1041844705139834) q[4];
ry(-1.502026434934798) q[5];
rz(1.2450070754043112) q[5];
ry(1.0427733684273124) q[6];
rz(2.8766261971410363) q[6];
ry(-3.0206130484965876) q[7];
rz(1.6538730415538234) q[7];
ry(-3.0950818647213527) q[8];
rz(0.7537868059685557) q[8];
ry(1.1166434617346424) q[9];
rz(0.45737895362252684) q[9];
ry(-0.02059347843099779) q[10];
rz(0.04359651170694688) q[10];
ry(0.39227190663273964) q[11];
rz(-0.4289244029070393) q[11];
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
ry(-1.6817565982947151) q[0];
rz(1.4543798466722802) q[0];
ry(-0.8920413873009547) q[1];
rz(2.9934216265761666) q[1];
ry(0.013424497788251869) q[2];
rz(-0.2521537563858516) q[2];
ry(3.074027963917869) q[3];
rz(3.0233744486234317) q[3];
ry(3.096746924997058) q[4];
rz(-2.6233449418117343) q[4];
ry(0.11740542790364827) q[5];
rz(1.6442776642684958) q[5];
ry(-1.1457322640023548) q[6];
rz(-0.19421425224781658) q[6];
ry(-2.8510991534942463) q[7];
rz(0.46830886528354376) q[7];
ry(2.613497475023443) q[8];
rz(2.5057112779814448) q[8];
ry(-2.1168373715528785) q[9];
rz(0.3446022009201313) q[9];
ry(0.7385446460910217) q[10];
rz(-1.5154860390316562) q[10];
ry(0.7497527708399341) q[11];
rz(-2.636865839190745) q[11];
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
ry(1.9760379033343825) q[0];
rz(-1.5294097778448632) q[0];
ry(0.714100587334443) q[1];
rz(-2.0591515772263467) q[1];
ry(0.059918216397863766) q[2];
rz(-2.681461436616969) q[2];
ry(-2.996720777328199) q[3];
rz(-0.8173635048925546) q[3];
ry(-0.5739065452834584) q[4];
rz(-3.083511885109938) q[4];
ry(-2.382522233590165) q[5];
rz(2.971688586462508) q[5];
ry(1.0732304879106045) q[6];
rz(-2.982350914502932) q[6];
ry(-0.04215310414900131) q[7];
rz(0.18768090256401315) q[7];
ry(2.057589649805613) q[8];
rz(-2.6390789001952206) q[8];
ry(-0.015894561819532423) q[9];
rz(2.7503325310066637) q[9];
ry(-2.868633929725265) q[10];
rz(-0.5546918917252187) q[10];
ry(-1.0578183020824579) q[11];
rz(-2.6644355350715254) q[11];
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
ry(2.216239085353757) q[0];
rz(-2.0454942152023303) q[0];
ry(-0.2546835713711854) q[1];
rz(1.2085019107709452) q[1];
ry(-3.140128006006958) q[2];
rz(0.1907368481223779) q[2];
ry(-1.529614609609724) q[3];
rz(2.270241542053958) q[3];
ry(-0.32345308686675894) q[4];
rz(-2.9955612200321045) q[4];
ry(1.065752355791333) q[5];
rz(-2.7565993754970957) q[5];
ry(-1.6709381807004113) q[6];
rz(2.9022411945180604) q[6];
ry(1.6706608569836527) q[7];
rz(0.6171958545843106) q[7];
ry(0.23870869469052203) q[8];
rz(2.989958478310372) q[8];
ry(0.0002909924833565114) q[9];
rz(-0.8697959065364405) q[9];
ry(-2.402273524903919) q[10];
rz(-0.17928908949191594) q[10];
ry(0.005763269229311695) q[11];
rz(1.4294184272774189) q[11];
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
ry(1.3409006739686828) q[0];
rz(1.4331446778532362) q[0];
ry(-0.8336713273430826) q[1];
rz(-2.7803452905308927) q[1];
ry(0.3702794149103976) q[2];
rz(0.107170017211061) q[2];
ry(-1.1107902151879232) q[3];
rz(-2.0568782193771966) q[3];
ry(2.8398193379615186) q[4];
rz(-2.232566685367387) q[4];
ry(0.18687291507375825) q[5];
rz(2.1742375846028423) q[5];
ry(3.117146124187523) q[6];
rz(1.4398721156252314) q[6];
ry(-0.8016929844307601) q[7];
rz(2.380557192623538) q[7];
ry(0.5457868868200042) q[8];
rz(0.09903505121307267) q[8];
ry(0.0011706955279488032) q[9];
rz(2.1140280910194296) q[9];
ry(2.7472178126052196) q[10];
rz(-0.2792048132290086) q[10];
ry(-0.8201430990710425) q[11];
rz(0.20407478539954155) q[11];
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
ry(2.0196416641815116) q[0];
rz(-1.7200079598418814) q[0];
ry(2.993537675130082) q[1];
rz(-2.1309864930160254) q[1];
ry(3.127916829140476) q[2];
rz(-1.5157690480668817) q[2];
ry(3.0560569529427455) q[3];
rz(-1.6977251463323009) q[3];
ry(-0.6403624369320644) q[4];
rz(-2.0719392848522764) q[4];
ry(-2.2844489476226744) q[5];
rz(1.2452414837512824) q[5];
ry(0.06617222943146608) q[6];
rz(1.534900155916043) q[6];
ry(-2.598999418788398) q[7];
rz(-0.9143137819929076) q[7];
ry(-3.0580520288440933) q[8];
rz(-0.43722073278575474) q[8];
ry(-3.00320562041415) q[9];
rz(-2.073872728733579) q[9];
ry(2.217254698700142) q[10];
rz(0.7413889414421985) q[10];
ry(-0.32682096892549134) q[11];
rz(2.5139770592932047) q[11];
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
ry(0.623295427071918) q[0];
rz(-1.4393839400142259) q[0];
ry(1.3551054295526035) q[1];
rz(-3.1220015448371585) q[1];
ry(-0.6907491157290284) q[2];
rz(0.8119974051592368) q[2];
ry(2.2106688487489925) q[3];
rz(-1.691659101928015) q[3];
ry(1.8959213164690034) q[4];
rz(-2.4109505672170597) q[4];
ry(-1.5090573666807847) q[5];
rz(-3.0814729235703484) q[5];
ry(0.007102223164623926) q[6];
rz(1.3575718749117076) q[6];
ry(-2.8960440227617896) q[7];
rz(-1.7027031619173503) q[7];
ry(2.5027746790859404) q[8];
rz(2.5952925520982717) q[8];
ry(-0.0920342585301599) q[9];
rz(2.2460392051501294) q[9];
ry(-2.9820542025373737) q[10];
rz(0.7202767160135845) q[10];
ry(0.7459111892407) q[11];
rz(-1.1976907835910389) q[11];
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
ry(-3.0232604378380743) q[0];
rz(1.796242625876343) q[0];
ry(-0.8478982500042871) q[1];
rz(-1.8640999325317855) q[1];
ry(0.3649898533706919) q[2];
rz(2.715503207331392) q[2];
ry(-3.09357039954672) q[3];
rz(0.21171580686265382) q[3];
ry(2.824713678469493) q[4];
rz(2.6632883982565874) q[4];
ry(-2.1597978661973594) q[5];
rz(2.7507033966561543) q[5];
ry(-2.753802932459056) q[6];
rz(0.93265795176429) q[6];
ry(-2.354661700687832) q[7];
rz(-1.0660426550726045) q[7];
ry(-2.751755220074685) q[8];
rz(1.8942089524601542) q[8];
ry(-3.076338092341824) q[9];
rz(-2.309048869250754) q[9];
ry(2.8013415360155642) q[10];
rz(0.8162146265644834) q[10];
ry(1.252771082166836) q[11];
rz(1.11375448148737) q[11];
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
ry(-2.4417231078931185) q[0];
rz(0.40551433657663516) q[0];
ry(-1.621685444381657) q[1];
rz(0.30838566288295244) q[1];
ry(-1.4517062156293044) q[2];
rz(-3.099981343542789) q[2];
ry(-0.7124689054048571) q[3];
rz(0.7966992280399207) q[3];
ry(-1.2488229300681561) q[4];
rz(-0.1439419365686316) q[4];
ry(-3.038634998134652) q[5];
rz(-2.6947901634554756) q[5];
ry(-3.1324438283762834) q[6];
rz(-0.16012434361153596) q[6];
ry(0.42910614398197744) q[7];
rz(0.5753730816439186) q[7];
ry(-0.013344975518235602) q[8];
rz(-1.2929187726723748) q[8];
ry(2.317269826980045) q[9];
rz(0.384314329718677) q[9];
ry(0.20626540235810736) q[10];
rz(-2.8084623997938096) q[10];
ry(-2.0715424220871874) q[11];
rz(-2.211538276648774) q[11];
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
ry(-3.1163940487639423) q[0];
rz(0.7478290598788995) q[0];
ry(3.1317988425805114) q[1];
rz(-2.4705795812688347) q[1];
ry(1.6394101387341469) q[2];
rz(-0.5456035634686273) q[2];
ry(2.022246007007384) q[3];
rz(-1.719329731457055) q[3];
ry(2.246523426276588) q[4];
rz(1.1363613211110213) q[4];
ry(-2.7564225296504055) q[5];
rz(0.32946204311487465) q[5];
ry(-1.596295367681166) q[6];
rz(-1.9342603409224592) q[6];
ry(-1.7643066091333282) q[7];
rz(3.114853891526562) q[7];
ry(2.9331747122792686) q[8];
rz(-0.270184227417724) q[8];
ry(3.121711299240299) q[9];
rz(0.48249058826098334) q[9];
ry(3.078855458074395) q[10];
rz(3.081733603514093) q[10];
ry(0.4104755929988979) q[11];
rz(1.3728204248628628) q[11];
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
ry(0.598364177245772) q[0];
rz(-2.41635651551517) q[0];
ry(0.31099288199330577) q[1];
rz(2.809418372793241) q[1];
ry(3.1375746882129514) q[2];
rz(2.541206055003433) q[2];
ry(0.02993649086928407) q[3];
rz(-1.4189623027901472) q[3];
ry(3.1414878147500094) q[4];
rz(-1.005573244326424) q[4];
ry(3.0873811715870847) q[5];
rz(1.0823436450996122) q[5];
ry(-3.07669579318138) q[6];
rz(-0.8590146885301015) q[6];
ry(-0.7134513596975509) q[7];
rz(-2.393632902147418) q[7];
ry(2.2899586379960004) q[8];
rz(-2.1708669599589094) q[8];
ry(-2.5463041891553377) q[9];
rz(2.1798473911755027) q[9];
ry(-2.787840475057363) q[10];
rz(0.29066898547579867) q[10];
ry(-0.45871191208907547) q[11];
rz(-1.6443083551749669) q[11];
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
ry(-0.06544308000518717) q[0];
rz(-3.120238655260782) q[0];
ry(-0.9644813381367231) q[1];
rz(-0.9073908476296424) q[1];
ry(-1.5261579776663385) q[2];
rz(1.7273295073639812) q[2];
ry(-1.1486503595507012) q[3];
rz(1.727948920627611) q[3];
ry(0.5668981819342781) q[4];
rz(1.7781642798998865) q[4];
ry(1.3654956981251694) q[5];
rz(1.6052229848314645) q[5];
ry(-0.15033299732794256) q[6];
rz(-1.4522939581524503) q[6];
ry(-0.0881954507161602) q[7];
rz(1.0942923778874225) q[7];
ry(1.716490662357903) q[8];
rz(2.2177150441570603) q[8];
ry(0.04378252805912392) q[9];
rz(0.799143309181065) q[9];
ry(2.8390813008585862) q[10];
rz(-0.5292062568147218) q[10];
ry(1.4634080230371564) q[11];
rz(1.7648905784510251) q[11];
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
ry(3.040242659616807) q[0];
rz(-3.1028938121077236) q[0];
ry(-0.6003583411146272) q[1];
rz(2.9893959517618525) q[1];
ry(-1.349274607224325) q[2];
rz(-0.9433781977777533) q[2];
ry(-2.1343824557579785) q[3];
rz(-0.37712617187995523) q[3];
ry(1.5243193629800602) q[4];
rz(-2.632179597749034) q[4];
ry(-3.132477156041326) q[5];
rz(2.98080187716553) q[5];
ry(0.012963173941820717) q[6];
rz(3.1075996115372297) q[6];
ry(1.6999080950881935) q[7];
rz(1.612246048980544) q[7];
ry(-2.3515121969350434) q[8];
rz(2.62964414301246) q[8];
ry(-0.000660801473667612) q[9];
rz(-1.735087302683039) q[9];
ry(2.363211406865214) q[10];
rz(2.96202781940919) q[10];
ry(1.1020631897523714) q[11];
rz(0.10401426048340667) q[11];
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
ry(2.535584776739465) q[0];
rz(0.46494622124046664) q[0];
ry(-2.6662068388419495) q[1];
rz(0.39420550913345487) q[1];
ry(0.08880566108103415) q[2];
rz(-0.28504537463461244) q[2];
ry(-0.10590952769493149) q[3];
rz(1.1637271105881508) q[3];
ry(-0.39071923892895377) q[4];
rz(-2.1007896532468577) q[4];
ry(1.999991567771798) q[5];
rz(-1.19562929473928) q[5];
ry(0.4572116236218378) q[6];
rz(-1.9612492913513222) q[6];
ry(0.9813510337939568) q[7];
rz(-0.05872370873111255) q[7];
ry(2.426693097482269) q[8];
rz(-1.2224843805823529) q[8];
ry(-0.12676646425615345) q[9];
rz(-1.6421889781736168) q[9];
ry(-2.9520241654449166) q[10];
rz(0.08258048217105785) q[10];
ry(-1.472864370538689) q[11];
rz(-0.5118214525280643) q[11];
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
ry(-2.950018960961872) q[0];
rz(2.797646551457425) q[0];
ry(-3.132176027090498) q[1];
rz(2.4002187817132263) q[1];
ry(-1.81859329441217) q[2];
rz(-0.5033829084799016) q[2];
ry(0.7963858740851197) q[3];
rz(-1.1573948477700262) q[3];
ry(-1.3915169501284135) q[4];
rz(-2.053942167007112) q[4];
ry(-3.1286274236429747) q[5];
rz(2.4432402467628913) q[5];
ry(-2.913371914758421) q[6];
rz(-1.9419494390932999) q[6];
ry(-1.6881224244064668) q[7];
rz(1.6841789138078918) q[7];
ry(0.2554002062106388) q[8];
rz(1.494404322612058) q[8];
ry(-3.106602474923808) q[9];
rz(-2.5946095826822364) q[9];
ry(-0.21055486132493953) q[10];
rz(2.3440221101982495) q[10];
ry(0.8580085838614532) q[11];
rz(0.6624067556734374) q[11];
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
ry(0.6477321112621377) q[0];
rz(1.9649908463161352) q[0];
ry(-0.6655043395414069) q[1];
rz(-2.545708051516474) q[1];
ry(-1.773453197017113) q[2];
rz(-0.8181486207613887) q[2];
ry(3.1047829335857955) q[3];
rz(1.1258223088930213) q[3];
ry(-2.99425332668248) q[4];
rz(1.2688228986023118) q[4];
ry(3.040423527947871) q[5];
rz(-0.05201292241097547) q[5];
ry(0.85833430325303) q[6];
rz(1.3965668815867485) q[6];
ry(-0.26955826408116296) q[7];
rz(0.4899445358685171) q[7];
ry(1.7077841740081352) q[8];
rz(1.0772833533825237) q[8];
ry(-0.6469629652122939) q[9];
rz(-2.6107637051254957) q[9];
ry(1.5263995477371877) q[10];
rz(2.2343189861648627) q[10];
ry(-1.7297081410996358) q[11];
rz(-1.1952277092540216) q[11];
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
ry(-1.1006190828248028) q[0];
rz(-1.4719870490430762) q[0];
ry(0.12954779293500615) q[1];
rz(1.578820491454597) q[1];
ry(2.647552587526925) q[2];
rz(-1.3713094629638203) q[2];
ry(3.0017621026425556) q[3];
rz(-0.6429832021119575) q[3];
ry(1.4285595114000422) q[4];
rz(1.7663277181839394) q[4];
ry(0.0310748665932854) q[5];
rz(2.740892707336927) q[5];
ry(-0.3433487505166486) q[6];
rz(-1.9525976358649493) q[6];
ry(1.1197631030420379) q[7];
rz(0.8656883149554785) q[7];
ry(-0.01208874780716851) q[8];
rz(-0.1301552393942211) q[8];
ry(1.9045873442332615) q[9];
rz(3.140496856209714) q[9];
ry(-0.26419802529852454) q[10];
rz(2.354717046223328) q[10];
ry(0.004376925186685092) q[11];
rz(0.5142414354749244) q[11];
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
ry(1.91138079704946) q[0];
rz(1.2878427455356996) q[0];
ry(-0.03801156802004968) q[1];
rz(-2.3986136285492448) q[1];
ry(-2.5371799447100836) q[2];
rz(-2.3727591615807198) q[2];
ry(1.965058633336243) q[3];
rz(1.675160222866689) q[3];
ry(-2.637277015600676) q[4];
rz(0.5810802046105197) q[4];
ry(1.769408529871041) q[5];
rz(-1.4876725349899722) q[5];
ry(0.12010066645222219) q[6];
rz(0.15633749484297432) q[6];
ry(0.06915239192803746) q[7];
rz(-1.1587217239579064) q[7];
ry(0.09004860541098182) q[8];
rz(1.3604730893486687) q[8];
ry(0.4326147298217285) q[9];
rz(-3.13497721665005) q[9];
ry(0.10995591547060268) q[10];
rz(-1.9278062526884394) q[10];
ry(-0.03580645721038955) q[11];
rz(2.673819655492281) q[11];
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
ry(1.3001013420949923) q[0];
rz(2.143698946283168) q[0];
ry(-2.9837231387672642) q[1];
rz(1.1605157445755676) q[1];
ry(-2.1235871897181617) q[2];
rz(3.127169068038157) q[2];
ry(-0.10456882127092555) q[3];
rz(-1.6132752021158516) q[3];
ry(-3.140051419402981) q[4];
rz(-0.20959957922529424) q[4];
ry(3.055568569991864) q[5];
rz(-2.9597047722853627) q[5];
ry(-0.04226471051553648) q[6];
rz(0.6891337798529947) q[6];
ry(-1.0593450874978816) q[7];
rz(0.18739605041547683) q[7];
ry(0.15821553422657253) q[8];
rz(1.8994821681840452) q[8];
ry(-1.9293110912991) q[9];
rz(0.022886527382413615) q[9];
ry(-2.353426814454903) q[10];
rz(0.26028121994536063) q[10];
ry(2.665483622512197) q[11];
rz(2.940569104617746) q[11];
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
ry(-0.5992919578174005) q[0];
rz(-2.316949394354969) q[0];
ry(-0.1990379247488124) q[1];
rz(-1.200668017537473) q[1];
ry(1.170537878602163) q[2];
rz(1.5006472458758169) q[2];
ry(-1.3898283352058867) q[3];
rz(-2.4067994386559812) q[3];
ry(1.484612885672578) q[4];
rz(0.22813949048269053) q[4];
ry(0.09838842604534914) q[5];
rz(-1.7218844813638006) q[5];
ry(-2.8469299778260817) q[6];
rz(0.9989915300918016) q[6];
ry(3.083521417052045) q[7];
rz(-2.2221313593289524) q[7];
ry(1.9241249550646427) q[8];
rz(0.08037284778155929) q[8];
ry(-2.0141045328790748) q[9];
rz(-2.7725670956961594) q[9];
ry(3.000033352910629) q[10];
rz(2.992113165048413) q[10];
ry(-1.4768286105110064) q[11];
rz(0.9483965576736343) q[11];
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
ry(0.12509042935969225) q[0];
rz(0.40973485660004844) q[0];
ry(-1.8715893186948462) q[1];
rz(0.12218804946909971) q[1];
ry(0.5561823185180632) q[2];
rz(0.17044882364525135) q[2];
ry(-3.0771936751781417) q[3];
rz(-1.642216206287536) q[3];
ry(3.117594319142613) q[4];
rz(2.6190403551518666) q[4];
ry(-0.11797493611830845) q[5];
rz(-2.750917175489911) q[5];
ry(1.4964539605334335) q[6];
rz(1.0061899795352465) q[6];
ry(2.04895506077166) q[7];
rz(-0.6302342854503307) q[7];
ry(-0.0928666168511798) q[8];
rz(1.476073836748065) q[8];
ry(0.007847426077153052) q[9];
rz(-0.7337397353903257) q[9];
ry(-3.1411522739581113) q[10];
rz(-0.1727508533767087) q[10];
ry(0.6709416495799321) q[11];
rz(-1.0001574416821342) q[11];
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
ry(-3.0472806539895205) q[0];
rz(2.1538701380642484) q[0];
ry(-0.21282289009901895) q[1];
rz(1.988211201335866) q[1];
ry(1.185643841878556) q[2];
rz(0.1842566370878752) q[2];
ry(3.00334488792697) q[3];
rz(2.0701261355659177) q[3];
ry(-0.31690014828485674) q[4];
rz(2.4289399320084764) q[4];
ry(-1.663732244183472) q[5];
rz(2.767583197947093) q[5];
ry(-2.0953684964427) q[6];
rz(0.42292695325844976) q[6];
ry(-0.372208474781556) q[7];
rz(1.5132629353726978) q[7];
ry(2.2393980829229596) q[8];
rz(1.7513686596703033) q[8];
ry(2.030728554519193) q[9];
rz(0.798373255043278) q[9];
ry(0.2439917038442161) q[10];
rz(-2.972634739757272) q[10];
ry(1.4148636114707787) q[11];
rz(-0.47357339911998414) q[11];
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
ry(-2.7296523748753088) q[0];
rz(2.0118132334377696) q[0];
ry(-2.6285878419378395) q[1];
rz(1.9860533352155096) q[1];
ry(3.003804792603074) q[2];
rz(0.9153868141250131) q[2];
ry(-3.0773900095398874) q[3];
rz(-2.4123602887527613) q[3];
ry(-0.02487979818338903) q[4];
rz(3.079860450828952) q[4];
ry(0.0029139245748001784) q[5];
rz(0.8431405584069624) q[5];
ry(0.06275408878561528) q[6];
rz(-1.7636007029860865) q[6];
ry(-0.007468243448169265) q[7];
rz(1.8618209626206506) q[7];
ry(-2.9324528440517064) q[8];
rz(0.049014051597401576) q[8];
ry(2.493838227563806) q[9];
rz(1.1101940644880575) q[9];
ry(-2.023222034972846) q[10];
rz(2.965164033929695) q[10];
ry(-0.12463187564625958) q[11];
rz(-2.9116697483909992) q[11];
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
ry(2.698851984928956) q[0];
rz(-0.54150881384715) q[0];
ry(3.0245532924793963) q[1];
rz(1.6063457710961622) q[1];
ry(1.9366810601861186) q[2];
rz(1.32034259127001) q[2];
ry(1.8681095613515506) q[3];
rz(-0.05455875338744675) q[3];
ry(-1.713615234297596) q[4];
rz(-0.47256071383849513) q[4];
ry(2.386674387123427) q[5];
rz(-2.510781918232258) q[5];
ry(-0.8803544982979616) q[6];
rz(-1.3394755029087482) q[6];
ry(-1.236551677349528) q[7];
rz(2.848404769766652) q[7];
ry(-1.884825126533669) q[8];
rz(-0.5326149301298112) q[8];
ry(-3.1125035397822303) q[9];
rz(2.4738995607909438) q[9];
ry(1.5686958918782405) q[10];
rz(0.5811138399216239) q[10];
ry(2.752712566049238) q[11];
rz(-1.2030548586691328) q[11];
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
ry(-0.5827910191896004) q[0];
rz(-1.7983149647460177) q[0];
ry(3.130080916405508) q[1];
rz(-1.2950811404329887) q[1];
ry(0.3823314143154803) q[2];
rz(-2.5177708841934674) q[2];
ry(0.14783601304917224) q[3];
rz(-3.030788098554194) q[3];
ry(-3.0637602822301817) q[4];
rz(0.18065222815799217) q[4];
ry(-0.1970504375520017) q[5];
rz(-0.11329595583096409) q[5];
ry(1.756136378273643) q[6];
rz(-0.42151807937057567) q[6];
ry(-0.023870854305852562) q[7];
rz(-1.3907772583918625) q[7];
ry(0.07745817740934431) q[8];
rz(2.050793068876493) q[8];
ry(1.9840217648167453) q[9];
rz(2.6381179566035313) q[9];
ry(0.23102996345652738) q[10];
rz(-2.363391028710696) q[10];
ry(-2.8832229021885327) q[11];
rz(-1.6968595735906171) q[11];
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
ry(1.520695486178846) q[0];
rz(0.7306998770609873) q[0];
ry(1.7615600726099245) q[1];
rz(-1.8713220492217042) q[1];
ry(2.182581915937499) q[2];
rz(-0.266580463220367) q[2];
ry(-2.5074722616718543) q[3];
rz(0.14707443099467937) q[3];
ry(-2.5208671088025216) q[4];
rz(0.30652206788288705) q[4];
ry(2.9177331316943214) q[5];
rz(2.7321367457677983) q[5];
ry(-0.049359059207235594) q[6];
rz(-2.71817564351805) q[6];
ry(0.006500851827165392) q[7];
rz(2.600677123993495) q[7];
ry(-3.127416945649128) q[8];
rz(-1.6884016038034986) q[8];
ry(0.00345764063898367) q[9];
rz(-0.43763788626288225) q[9];
ry(-2.6957403916932847) q[10];
rz(-2.58665040437305) q[10];
ry(1.4206269965098821) q[11];
rz(-0.5871548900413596) q[11];
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
ry(0.6021582693920277) q[0];
rz(-0.6606324946228455) q[0];
ry(-0.018260819689446685) q[1];
rz(2.2996691810285865) q[1];
ry(-0.6898222110943069) q[2];
rz(0.18930872975282906) q[2];
ry(-3.000393617906551) q[3];
rz(-1.8032129752419246) q[3];
ry(-0.07902593436228766) q[4];
rz(-1.702667234996153) q[4];
ry(0.0365812874079845) q[5];
rz(1.6330536004763763) q[5];
ry(-1.7011698356526161) q[6];
rz(-1.8705934678018663) q[6];
ry(0.008246275759679391) q[7];
rz(-1.5938718491345294) q[7];
ry(-1.5768747602498057) q[8];
rz(-0.04063614205695416) q[8];
ry(-1.2592846497093946) q[9];
rz(-1.294449587288864) q[9];
ry(0.03776933147004885) q[10];
rz(-2.478616874199913) q[10];
ry(-1.3995852161748972) q[11];
rz(-0.2178634711902241) q[11];
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
ry(1.9804454314828117) q[0];
rz(2.1931505477884317) q[0];
ry(-1.0729747551655948) q[1];
rz(-1.040794824189616) q[1];
ry(-0.24954510874888045) q[2];
rz(-1.011658598385723) q[2];
ry(1.5320315934558018) q[3];
rz(-1.5372904983956905) q[3];
ry(1.5620024352718984) q[4];
rz(2.3715212520935927) q[4];
ry(1.3498230244756266) q[5];
rz(-1.497579581059404) q[5];
ry(0.19768571636026142) q[6];
rz(-1.5117507574729991) q[6];
ry(0.2947583449099842) q[7];
rz(2.4292486271786378) q[7];
ry(-2.758896071381002) q[8];
rz(-0.08004169650645206) q[8];
ry(-1.5697400160992814) q[9];
rz(-3.1380849757280886) q[9];
ry(2.4987835986884455) q[10];
rz(0.15800890109580304) q[10];
ry(-0.9804935459716511) q[11];
rz(-0.3497682528302457) q[11];
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
ry(0.45211837535675076) q[0];
rz(2.6390379366095154) q[0];
ry(3.1360056917617243) q[1];
rz(-2.5385346321069475) q[1];
ry(-0.08381033711324104) q[2];
rz(2.3685832849696977) q[2];
ry(3.1044644041413334) q[3];
rz(0.488921608186617) q[3];
ry(3.141192358801495) q[4];
rz(1.6789226827068322) q[4];
ry(0.07005930561364959) q[5];
rz(2.4047526367688015) q[5];
ry(-0.01740376723065809) q[6];
rz(1.8984018085820642) q[6];
ry(3.1405714697743843) q[7];
rz(-2.8389806905310206) q[7];
ry(3.1292235787945373) q[8];
rz(1.5626248655512986) q[8];
ry(-0.019194327935215583) q[9];
rz(1.5691390544280124) q[9];
ry(-1.572308282984704) q[10];
rz(-1.570653809886495) q[10];
ry(-1.4024102790649586) q[11];
rz(-2.692404779025787) q[11];
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
ry(-2.925407671315298) q[0];
rz(1.123230887197641) q[0];
ry(-1.4201720032141065) q[1];
rz(0.7842035610510889) q[1];
ry(1.6371378164055583) q[2];
rz(-2.254253767641516) q[2];
ry(-2.8494936178864934) q[3];
rz(-1.576215557001292) q[3];
ry(-0.0659133548490507) q[4];
rz(2.455847582414634) q[4];
ry(-0.3873098311354007) q[5];
rz(-2.8337976480295244) q[5];
ry(1.7262252767404656) q[6];
rz(2.6630506684340287) q[6];
ry(1.7369425375326921) q[7];
rz(-2.0765740086814555) q[7];
ry(-1.5948036801976058) q[8];
rz(1.4948504340033368) q[8];
ry(-1.5705161955955187) q[9];
rz(0.2699766995594478) q[9];
ry(-1.5721485012925114) q[10];
rz(-2.9788162993018292) q[10];
ry(3.141551797848223) q[11];
rz(2.0436296878186244) q[11];