OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.20290554765377333) q[0];
rz(0.6329797025859678) q[0];
ry(0.30316110751735176) q[1];
rz(1.1115152937447625) q[1];
ry(-2.5767600057916042) q[2];
rz(1.6896515688450127) q[2];
ry(-0.39112390586413737) q[3];
rz(-1.1157796856217095) q[3];
ry(-2.183452547579523) q[4];
rz(1.4655535313657015) q[4];
ry(-0.9470281758833333) q[5];
rz(2.736935351719827) q[5];
ry(2.7854600856798073) q[6];
rz(2.69999251691362) q[6];
ry(3.1308453890633037) q[7];
rz(2.5397969365737563) q[7];
ry(0.6948633829393586) q[8];
rz(-1.0079995610778916) q[8];
ry(1.423870361493659) q[9];
rz(-2.4723175851176475) q[9];
ry(1.4796343165115688) q[10];
rz(-3.0422327128226843) q[10];
ry(1.0443710406939317) q[11];
rz(-0.5437754368584661) q[11];
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
ry(2.602793298648672) q[0];
rz(1.9030364461887403) q[0];
ry(-1.3887670362948825) q[1];
rz(-2.450109864813165) q[1];
ry(0.8471933587093341) q[2];
rz(2.81995755265711) q[2];
ry(-1.3158716113243087) q[3];
rz(0.19561350994456375) q[3];
ry(0.01711833935832452) q[4];
rz(-0.5141127912670633) q[4];
ry(-2.605209856624568) q[5];
rz(0.22266856308586175) q[5];
ry(1.8804056749544733) q[6];
rz(-0.6332818506290314) q[6];
ry(-3.13673844438594) q[7];
rz(2.6978891024171445) q[7];
ry(-2.804799161299357) q[8];
rz(2.4491840014268633) q[8];
ry(2.32660295247393) q[9];
rz(2.372648386312964) q[9];
ry(-0.4309152184980629) q[10];
rz(2.8244368667646396) q[10];
ry(-0.646668037192197) q[11];
rz(-2.3414964182537714) q[11];
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
ry(-0.015155923899985881) q[0];
rz(2.4095569265617254) q[0];
ry(-2.497242034075082) q[1];
rz(-1.3410678326414418) q[1];
ry(-1.120693508501236) q[2];
rz(2.4548165474856107) q[2];
ry(-0.676213040465135) q[3];
rz(-2.404390616585382) q[3];
ry(-0.47860716977973894) q[4];
rz(1.6139826271371263) q[4];
ry(-2.496168827249732) q[5];
rz(2.453297727515015) q[5];
ry(-2.7690494522064504) q[6];
rz(2.010848309447768) q[6];
ry(3.1255408266264584) q[7];
rz(0.7827058804038183) q[7];
ry(-0.7211387618717864) q[8];
rz(0.4352383485344791) q[8];
ry(-2.113378839193558) q[9];
rz(-0.28474954058480856) q[9];
ry(3.0092118086389728) q[10];
rz(1.2383321056347742) q[10];
ry(3.104917950264205) q[11];
rz(1.2548032580912414) q[11];
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
ry(1.8588973629327699) q[0];
rz(-2.9170399034130536) q[0];
ry(-0.40989087549008296) q[1];
rz(0.5719965636585148) q[1];
ry(-1.5423985652697967) q[2];
rz(-2.1175887012298737) q[2];
ry(-2.334466790446799) q[3];
rz(1.8202017360238119) q[3];
ry(1.665245086695763) q[4];
rz(-2.5175320910163483) q[4];
ry(1.901223794714598) q[5];
rz(-2.3601273018546642) q[5];
ry(-0.9825674522054035) q[6];
rz(0.4643835647642147) q[6];
ry(-1.4360359105152911) q[7];
rz(-2.09777828319292) q[7];
ry(2.452150401035265) q[8];
rz(-1.1753180821798606) q[8];
ry(2.333943202217836) q[9];
rz(2.1128138307573394) q[9];
ry(-3.0851837372481445) q[10];
rz(3.001229730653719) q[10];
ry(-0.9611659961113264) q[11];
rz(1.7595222097341159) q[11];
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
ry(0.13554242918687276) q[0];
rz(-1.4572692522775652) q[0];
ry(-2.6147442602743505) q[1];
rz(1.761467697178733) q[1];
ry(-0.532979648520382) q[2];
rz(-0.4841302903132136) q[2];
ry(2.347464257392564) q[3];
rz(-1.8553529339267545) q[3];
ry(-0.15244146878965376) q[4];
rz(-1.8912181984858327) q[4];
ry(2.977804292205445) q[5];
rz(1.676287710627957) q[5];
ry(-3.1392574475465698) q[6];
rz(0.3400887889048842) q[6];
ry(3.1405697362613605) q[7];
rz(3.101447460389718) q[7];
ry(-0.9667919978057286) q[8];
rz(-3.060055511178599) q[8];
ry(1.601255090440299) q[9];
rz(2.7118664343030123) q[9];
ry(-1.9432392280379234) q[10];
rz(-0.4274747846848194) q[10];
ry(-1.0952226603371085) q[11];
rz(-3.12740818126932) q[11];
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
ry(-2.5821449975520507) q[0];
rz(0.6581444036631074) q[0];
ry(1.9138590120225816) q[1];
rz(0.045976334222674164) q[1];
ry(0.9659325429490675) q[2];
rz(-0.979592666598256) q[2];
ry(-2.2715067397149795) q[3];
rz(-2.1919786925635547) q[3];
ry(-2.6860830436249716) q[4];
rz(-2.172570722420339) q[4];
ry(2.535917372930737) q[5];
rz(1.381660564068702) q[5];
ry(2.209911631052796) q[6];
rz(-1.3387793728857744) q[6];
ry(1.5826953498512868) q[7];
rz(0.20393694876285645) q[7];
ry(0.23605076152157253) q[8];
rz(-2.3586118440483466) q[8];
ry(0.6761077180154604) q[9];
rz(-2.929992071539112) q[9];
ry(1.2773457116133446) q[10];
rz(1.3320277957492594) q[10];
ry(0.34884057211979247) q[11];
rz(0.7804320793122616) q[11];
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
ry(-1.4885109115478126) q[0];
rz(0.4720968610767546) q[0];
ry(1.2743949925327773) q[1];
rz(-0.25646665002827707) q[1];
ry(0.3456211427700578) q[2];
rz(0.1406464744249381) q[2];
ry(1.6358252867239607) q[3];
rz(-0.2824763035311389) q[3];
ry(-1.5547944935568103) q[4];
rz(-2.192988970715395) q[4];
ry(-2.9027912432177856) q[5];
rz(1.0924361591032445) q[5];
ry(0.001762992037542155) q[6];
rz(1.2450327149261637) q[6];
ry(0.039426911566025964) q[7];
rz(-1.4315551910555628) q[7];
ry(-1.1475113419429288) q[8];
rz(-1.3255288122885176) q[8];
ry(1.533471239969998) q[9];
rz(0.7786972146051384) q[9];
ry(-2.173218419644758) q[10];
rz(2.6224049511524665) q[10];
ry(-0.29262919928993947) q[11];
rz(-2.9627909771479453) q[11];
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
ry(1.421793422372162) q[0];
rz(-3.034014312456341) q[0];
ry(-0.27035758710241176) q[1];
rz(0.37874833247817724) q[1];
ry(-2.1505614838480973) q[2];
rz(0.5618921952112457) q[2];
ry(-0.04661412072456823) q[3];
rz(2.5137722539142775) q[3];
ry(-0.8311545408928369) q[4];
rz(-1.4625832502965483) q[4];
ry(1.60844858526842) q[5];
rz(0.2840166255148749) q[5];
ry(0.5661274000531683) q[6];
rz(-3.0528518343177824) q[6];
ry(-2.8123914173407685) q[7];
rz(-1.6530702255397507) q[7];
ry(-2.0618811600880873) q[8];
rz(-1.7417637641384662) q[8];
ry(1.280329556731115) q[9];
rz(-1.5732047072750923) q[9];
ry(3.099402191369163) q[10];
rz(0.5885269871041224) q[10];
ry(2.5955482222319275) q[11];
rz(1.694128460024639) q[11];
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
ry(-0.297448912212551) q[0];
rz(2.4632311557095523) q[0];
ry(2.7446748212472563) q[1];
rz(-1.3104926091622326) q[1];
ry(-3.0494995475144977) q[2];
rz(2.68367384568887) q[2];
ry(-0.9346080478289922) q[3];
rz(2.438225556648169) q[3];
ry(-0.5942350581238903) q[4];
rz(-3.038785667939214) q[4];
ry(-0.09413136071704464) q[5];
rz(1.1371268528169622) q[5];
ry(3.0521128875816217) q[6];
rz(-2.5277445351894223) q[6];
ry(0.04045070494782315) q[7];
rz(-2.3564699612386475) q[7];
ry(0.4793688465195227) q[8];
rz(2.3880682188118674) q[8];
ry(2.1358919475836684) q[9];
rz(-1.3548153234463802) q[9];
ry(-2.6064770919307154) q[10];
rz(-2.6304642385249966) q[10];
ry(-1.725570102059085) q[11];
rz(-1.3717804128579665) q[11];
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
ry(0.1915668728244135) q[0];
rz(-1.4948457330691873) q[0];
ry(-0.5761287376654227) q[1];
rz(2.0736987317852016) q[1];
ry(-3.0877415462793865) q[2];
rz(-0.9519790562006895) q[2];
ry(0.6278710040729267) q[3];
rz(-2.5438771009263346) q[3];
ry(1.411457232570931) q[4];
rz(-0.5212343561582763) q[4];
ry(0.07732522771369565) q[5];
rz(0.44866646049285835) q[5];
ry(0.5463763976610512) q[6];
rz(0.38940506348196374) q[6];
ry(1.255430689686964) q[7];
rz(-1.0189923697350345) q[7];
ry(-2.0196334212785483) q[8];
rz(-2.6803222124327335) q[8];
ry(2.7132657034639407) q[9];
rz(-1.0309219173885662) q[9];
ry(-1.0909650738168406) q[10];
rz(-0.5289967373439112) q[10];
ry(-3.0572552973965026) q[11];
rz(0.2256215599743694) q[11];
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
ry(-0.6545196787426424) q[0];
rz(-2.776765351985708) q[0];
ry(1.9518186656539445) q[1];
rz(-1.8413517274148996) q[1];
ry(1.6314930725485777) q[2];
rz(-1.1543530883779498) q[2];
ry(-0.6792161448880948) q[3];
rz(0.10239227880097435) q[3];
ry(-0.06594112030141178) q[4];
rz(-2.3773140225394322) q[4];
ry(3.0720102684004353) q[5];
rz(1.1746530464439338) q[5];
ry(-0.09885740361070144) q[6];
rz(-2.33636564855801) q[6];
ry(-0.009620262488917418) q[7];
rz(-1.2842833692756923) q[7];
ry(-1.4215581512717463) q[8];
rz(-0.6849218717235834) q[8];
ry(0.31843659938512037) q[9];
rz(2.4359349628303564) q[9];
ry(0.7162832932065228) q[10];
rz(-2.0934640959571507) q[10];
ry(1.8418522750109716) q[11];
rz(2.3855796016781974) q[11];
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
ry(0.36351458307540635) q[0];
rz(0.4358124959931605) q[0];
ry(2.5633432288595133) q[1];
rz(1.9262027063313265) q[1];
ry(1.935940646454554) q[2];
rz(1.2169803799347183) q[2];
ry(2.8760762234091697) q[3];
rz(0.5849676451984972) q[3];
ry(-3.1374447133709613) q[4];
rz(3.059811925771346) q[4];
ry(2.0882664297041735) q[5];
rz(2.2527419592643643) q[5];
ry(-1.0440817248379242) q[6];
rz(2.921421537072798) q[6];
ry(1.5375301577106553) q[7];
rz(-2.2480300037168206) q[7];
ry(2.0334838556738237) q[8];
rz(-1.1727053864918826) q[8];
ry(1.001304135882414) q[9];
rz(-1.1467205714176085) q[9];
ry(-0.6369465591585861) q[10];
rz(2.1550861788379914) q[10];
ry(-3.111323210829257) q[11];
rz(0.3344020808806152) q[11];
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
ry(-1.3153127387834342) q[0];
rz(3.043137708701906) q[0];
ry(-1.2782967360454291) q[1];
rz(1.0660537561497856) q[1];
ry(-3.098954059130411) q[2];
rz(-2.275161013415369) q[2];
ry(1.199767533060648) q[3];
rz(-0.7382327530073765) q[3];
ry(-3.0510708634817765) q[4];
rz(0.1322416811833316) q[4];
ry(0.011040398126385926) q[5];
rz(-2.136014326670939) q[5];
ry(-0.012713483713169715) q[6];
rz(-2.635161498195731) q[6];
ry(-0.015370162242205106) q[7];
rz(-0.19807264393542126) q[7];
ry(-1.742149680782851) q[8];
rz(-2.551786725510774) q[8];
ry(1.1192335225374128) q[9];
rz(1.103875143847592) q[9];
ry(2.4503788057099336) q[10];
rz(-1.602034301128276) q[10];
ry(-2.857669210967186) q[11];
rz(2.495421586272643) q[11];
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
ry(-2.5829736656847717) q[0];
rz(-0.659717350864306) q[0];
ry(-2.03121372954437) q[1];
rz(0.7150956443458499) q[1];
ry(1.5595601898252243) q[2];
rz(1.4290291777126365) q[2];
ry(-2.8828042694654137) q[3];
rz(2.593509490920204) q[3];
ry(2.389713157728551) q[4];
rz(0.42973466933325744) q[4];
ry(-1.2499527208122563) q[5];
rz(-2.1234041599777704) q[5];
ry(0.8754587612509283) q[6];
rz(2.345250866881212) q[6];
ry(2.313221467009711) q[7];
rz(1.3556002820542292) q[7];
ry(0.7444323489786067) q[8];
rz(-2.6738393502944686) q[8];
ry(-1.3543931389944943) q[9];
rz(1.1708004334862376) q[9];
ry(-0.6714308993220208) q[10];
rz(1.2178385813690111) q[10];
ry(-0.9864259062107428) q[11];
rz(-1.6773662817737511) q[11];
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
ry(-0.9410301020292706) q[0];
rz(-2.294824472272073) q[0];
ry(0.96195705621061) q[1];
rz(-3.1395920366820187) q[1];
ry(1.932954333645621) q[2];
rz(2.069852324574731) q[2];
ry(1.1760142324163292) q[3];
rz(0.36789991751819356) q[3];
ry(-0.578288460028917) q[4];
rz(1.6765550768553208) q[4];
ry(-3.0601252033955375) q[5];
rz(2.28657551283729) q[5];
ry(0.9151555247342759) q[6];
rz(-3.1264749607469566) q[6];
ry(-0.03912905367251884) q[7];
rz(0.5212026712156422) q[7];
ry(1.512549027493275) q[8];
rz(0.9497058797143254) q[8];
ry(-0.22869563322502065) q[9];
rz(2.1655704368090967) q[9];
ry(-2.712226671000691) q[10];
rz(-0.07553461380466153) q[10];
ry(1.8777131141649523) q[11];
rz(-2.2709526230987827) q[11];
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
ry(2.391617448434066) q[0];
rz(0.029715928263139044) q[0];
ry(-1.1326569496380872) q[1];
rz(0.6473663312775748) q[1];
ry(-0.6477567065288866) q[2];
rz(2.4122336092141126) q[2];
ry(0.5821739312328607) q[3];
rz(-1.883272720648436) q[3];
ry(-2.9889477761869547) q[4];
rz(0.5119383126574643) q[4];
ry(-0.02177549402370893) q[5];
rz(-2.682328372845419) q[5];
ry(2.8946475088042303) q[6];
rz(-1.8343879999805217) q[6];
ry(-1.970379616691777) q[7];
rz(3.113415297794573) q[7];
ry(-2.0078256403486394) q[8];
rz(-0.15821802774141513) q[8];
ry(-2.0351035059465357) q[9];
rz(-0.21641864244474623) q[9];
ry(-1.2273618901459287) q[10];
rz(-2.8371309495509727) q[10];
ry(2.0260267961733) q[11];
rz(2.972860546864384) q[11];
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
ry(-1.9183973573274864) q[0];
rz(0.4861087640076746) q[0];
ry(3.1220836929580806) q[1];
rz(1.5402766952061304) q[1];
ry(2.8647054995603725) q[2];
rz(-3.078041310357041) q[2];
ry(-1.6794664522396614) q[3];
rz(2.726442220554796) q[3];
ry(-2.949784131539941) q[4];
rz(-1.286205022716231) q[4];
ry(-0.09654839437750642) q[5];
rz(2.851029918942034) q[5];
ry(2.8782324600104983) q[6];
rz(-0.818366608757697) q[6];
ry(0.8114200609081009) q[7];
rz(-0.2009228091202253) q[7];
ry(-0.8475315219198425) q[8];
rz(-3.025519601641146) q[8];
ry(-2.389814164752865) q[9];
rz(-2.628456626976163) q[9];
ry(-1.5496537364855563) q[10];
rz(-1.9088427506464238) q[10];
ry(-1.5610841322542512) q[11];
rz(3.1145487624090014) q[11];
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
ry(-0.08111095795614866) q[0];
rz(3.1266709737439933) q[0];
ry(1.6916081584144278) q[1];
rz(-2.538227318757527) q[1];
ry(0.5409047981597985) q[2];
rz(2.9052814515116845) q[2];
ry(2.6455150117106285) q[3];
rz(-2.729128843086923) q[3];
ry(0.05420543761690469) q[4];
rz(-0.013134642710134871) q[4];
ry(-3.1230364224058786) q[5];
rz(-1.6947162621356746) q[5];
ry(-0.018051636213607836) q[6];
rz(-1.3915207651963744) q[6];
ry(3.1236843335090816) q[7];
rz(-1.4839023145678585) q[7];
ry(1.543102075692033) q[8];
rz(-0.004663993437534668) q[8];
ry(3.0482307868996674) q[9];
rz(3.1260068182572383) q[9];
ry(-2.3727974416783573) q[10];
rz(-2.1218197004025794) q[10];
ry(-1.5118380402143274) q[11];
rz(-0.0012906020344931723) q[11];
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
ry(2.118558760300342) q[0];
rz(-2.8711213171137993) q[0];
ry(1.5479620816705504) q[1];
rz(0.18367295506565193) q[1];
ry(-1.361836611154735) q[2];
rz(-0.42310905878231164) q[2];
ry(-0.07849451475383429) q[3];
rz(-1.1685134039579967) q[3];
ry(0.09242861482905429) q[4];
rz(0.9396718583840281) q[4];
ry(3.1401123551426497) q[5];
rz(1.8257616894857929) q[5];
ry(-0.6501226577190407) q[6];
rz(2.81767745189756) q[6];
ry(0.02174490751709122) q[7];
rz(-0.2527894639067007) q[7];
ry(0.8274921664439121) q[8];
rz(1.5775905532402144) q[8];
ry(-1.5355110652501924) q[9];
rz(-0.28253667796055043) q[9];
ry(-1.20788124133162) q[10];
rz(-2.6011576482572725) q[10];
ry(-0.7825196872160527) q[11];
rz(-0.7549061026287113) q[11];
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
ry(-1.53903207880367) q[0];
rz(1.1052447495560909) q[0];
ry(-0.8686854070134434) q[1];
rz(-0.9859574929700193) q[1];
ry(1.4127299725280873) q[2];
rz(1.6143880600314546) q[2];
ry(-0.2781537292770374) q[3];
rz(1.580267167275293) q[3];
ry(-3.011969031639151) q[4];
rz(0.9276399131404878) q[4];
ry(2.330571835527959) q[5];
rz(2.5251920129089203) q[5];
ry(-0.04383913499396407) q[6];
rz(2.5911583764047323) q[6];
ry(-1.4829812591882554) q[7];
rz(-2.004146234494338) q[7];
ry(1.606517145273414) q[8];
rz(-1.084813454326931) q[8];
ry(1.273884823653223) q[9];
rz(0.08669926002251394) q[9];
ry(1.649018696596202) q[10];
rz(2.253557569702064) q[10];
ry(1.917496006587137) q[11];
rz(0.3303568170889578) q[11];
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
ry(2.5474482581459172) q[0];
rz(1.703417719330913) q[0];
ry(2.5204874909368855) q[1];
rz(0.5270374248560875) q[1];
ry(2.721012702784003) q[2];
rz(-2.411025394227433) q[2];
ry(2.319290289465401) q[3];
rz(-2.837613521953067) q[3];
ry(3.123701192744002) q[4];
rz(-1.7308685454276664) q[4];
ry(-0.031147974982477677) q[5];
rz(2.8389720847394613) q[5];
ry(-3.1414063999396986) q[6];
rz(-0.10307108686237028) q[6];
ry(0.01849207936896935) q[7];
rz(1.2301588972165636) q[7];
ry(0.21838531089476482) q[8];
rz(1.0990251529934723) q[8];
ry(-0.9055452320467383) q[9];
rz(0.8558198611128409) q[9];
ry(-3.032890176394602) q[10];
rz(2.3850395293393074) q[10];
ry(1.971744364179637) q[11];
rz(1.3012044445292057) q[11];
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
ry(0.7781629231362202) q[0];
rz(2.93526160001751) q[0];
ry(2.0179564768761376) q[1];
rz(-1.5818853450456318) q[1];
ry(0.3453586420307194) q[2];
rz(1.5284525135986227) q[2];
ry(-0.5761176197681962) q[3];
rz(-0.815145950024438) q[3];
ry(0.5177566493397903) q[4];
rz(-0.6213975647293318) q[4];
ry(2.0002914213709277) q[5];
rz(-0.9256867816909846) q[5];
ry(3.1052202893679017) q[6];
rz(-1.9828657291810847) q[6];
ry(-3.0564030007622787) q[7];
rz(-0.12510397899670878) q[7];
ry(3.0886589525186454) q[8];
rz(-1.2988722359566984) q[8];
ry(2.8051786689068647) q[9];
rz(-2.1102886958433835) q[9];
ry(0.08775628569076924) q[10];
rz(3.082256978214475) q[10];
ry(2.938653445913804) q[11];
rz(-0.7466606966261677) q[11];
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
ry(2.508225991569255) q[0];
rz(-1.7983278723549212) q[0];
ry(0.7745985129001252) q[1];
rz(-0.9220299294867722) q[1];
ry(2.3431979632289384) q[2];
rz(1.0036767493310939) q[2];
ry(-2.719953846558785) q[3];
rz(-2.848117216864471) q[3];
ry(-3.12598564098798) q[4];
rz(0.05763551033872804) q[4];
ry(-0.018936860323915176) q[5];
rz(0.3129713253608689) q[5];
ry(-1.5732061369958756) q[6];
rz(1.1347370859218904) q[6];
ry(3.1151185638946437) q[7];
rz(-0.22904952708696943) q[7];
ry(-2.5186279408074386) q[8];
rz(2.353176317974981) q[8];
ry(-0.6800382137123293) q[9];
rz(1.195286147558029) q[9];
ry(1.8342511503061245) q[10];
rz(-2.9949976968435545) q[10];
ry(-2.6304871863965085) q[11];
rz(-0.1966598007756524) q[11];
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
ry(1.1565012112489927) q[0];
rz(2.546544898070498) q[0];
ry(2.970928268505624) q[1];
rz(1.8813463555947108) q[1];
ry(-1.9665887722150879) q[2];
rz(1.555282298706361) q[2];
ry(1.8635170838558883) q[3];
rz(1.2847472031558176) q[3];
ry(-2.5767040257202614) q[4];
rz(-1.2402298606410502) q[4];
ry(-1.5606431026708045) q[5];
rz(0.00437416798177897) q[5];
ry(-0.0018358450235904655) q[6];
rz(2.812715818664555) q[6];
ry(1.0545090015134972) q[7];
rz(-0.003481476015847124) q[7];
ry(2.7721525698170923) q[8];
rz(-2.8856076457416306) q[8];
ry(-2.6849402446424993) q[9];
rz(-0.702800962021116) q[9];
ry(-1.5770970978540266) q[10];
rz(-3.050288684167556) q[10];
ry(-0.5593830571172584) q[11];
rz(0.712237668190071) q[11];
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
ry(0.17406990065380956) q[0];
rz(1.0631673980935128) q[0];
ry(0.9890874857623507) q[1];
rz(0.5301640429439286) q[1];
ry(-0.7142768116890119) q[2];
rz(2.826566733610896) q[2];
ry(-0.5505655979711648) q[3];
rz(-1.9014203161817036) q[3];
ry(-1.5749513227092582) q[4];
rz(1.5720259792239313) q[4];
ry(0.010582018775131985) q[5];
rz(1.3855378286000464) q[5];
ry(-0.2926760362340064) q[6];
rz(2.3443865488521256) q[6];
ry(-1.4484300975584175) q[7];
rz(-0.006360772624728595) q[7];
ry(0.914704621273132) q[8];
rz(-2.0199536219509726) q[8];
ry(0.8236298898017673) q[9];
rz(-2.0139424774493597) q[9];
ry(0.5019349363071033) q[10];
rz(-1.9072032390953417) q[10];
ry(-3.113270414417961) q[11];
rz(-0.5640956354243567) q[11];
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
ry(3.058592190685927) q[0];
rz(0.10114435986130707) q[0];
ry(-2.7746797381968547) q[1];
rz(-0.9183570811558122) q[1];
ry(2.485739659280082) q[2];
rz(-0.7517387999976537) q[2];
ry(1.5724663816697824) q[3];
rz(-1.5703938831025681) q[3];
ry(1.5707815204491764) q[4];
rz(-2.33358864337976) q[4];
ry(-0.000987450014446532) q[5];
rz(-2.961249833847433) q[5];
ry(2.972013678732041) q[6];
rz(2.2152304655836073) q[6];
ry(-1.472905659535928) q[7];
rz(-3.139481727962839) q[7];
ry(-0.0009033367115164539) q[8];
rz(-1.5478749869699953) q[8];
ry(-0.0034695856298165846) q[9];
rz(-2.9870029175289377) q[9];
ry(3.0889093775313614) q[10];
rz(-2.371793347121393) q[10];
ry(1.770343086030301) q[11];
rz(-2.008747014023792) q[11];
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
ry(-2.3413695531393706) q[0];
rz(2.1898171160796642) q[0];
ry(-2.495394339234272) q[1];
rz(-2.3073098774169236) q[1];
ry(-0.9113181091633438) q[2];
rz(-0.00032311624617905184) q[2];
ry(-0.8821635890178973) q[3];
rz(-2.6209979306536546) q[3];
ry(-1.5713238801251024) q[4];
rz(1.5705169857991357) q[4];
ry(1.9543697980096786) q[5];
rz(2.690202132318192) q[5];
ry(2.7112491120004942) q[6];
rz(1.3786044271351852) q[6];
ry(1.261766679719701) q[7];
rz(0.007212301810822552) q[7];
ry(-0.08489580472063867) q[8];
rz(1.1698997251752283) q[8];
ry(-0.7338783041909789) q[9];
rz(1.41018012081648) q[9];
ry(2.147587303702627) q[10];
rz(-0.8821970192663698) q[10];
ry(-1.4068819194398792) q[11];
rz(-1.3537614659505541) q[11];
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
ry(-2.5618043128285914) q[0];
rz(-2.110049173556173) q[0];
ry(-3.1394063472422338) q[1];
rz(-1.2340483239232407) q[1];
ry(1.1982560558698143) q[2];
rz(0.5809998004275536) q[2];
ry(3.1406433264845144) q[3];
rz(-2.6208861801500487) q[3];
ry(-0.8389518126092659) q[4];
rz(-0.0009857312487507371) q[4];
ry(-3.141048769127074) q[5];
rz(2.689761073901428) q[5];
ry(-7.965618439299841e-05) q[6];
rz(2.7967643906554485) q[6];
ry(-2.755436730016547) q[7];
rz(-3.1389161493483635) q[7];
ry(-0.009041680154908251) q[8];
rz(-2.36303747584173) q[8];
ry(-3.140420162069786) q[9];
rz(-0.42709195635322317) q[9];
ry(0.02268162122787852) q[10];
rz(-0.14911019227535774) q[10];
ry(-2.9262316451964683) q[11];
rz(-0.7015314568689011) q[11];
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
ry(-2.079171831156385) q[0];
rz(-1.2669945438479502) q[0];
ry(-0.7825995753797931) q[1];
rz(2.9126036549326857) q[1];
ry(-0.747970787296695) q[2];
rz(2.6926552404598416) q[2];
ry(1.6056573395641651) q[3];
rz(-0.958467872484524) q[3];
ry(1.916740598686534) q[4];
rz(-1.5707813333240006) q[4];
ry(1.0146192310679625) q[5];
rz(1.339566064993977) q[5];
ry(0.0853426242672457) q[6];
rz(1.285779561292635) q[6];
ry(-0.19141690075842432) q[7];
rz(-1.1332559064379888) q[7];
ry(-1.4042569945432835) q[8];
rz(1.9434350141995134) q[8];
ry(-2.495083393906777) q[9];
rz(0.5750482715007204) q[9];
ry(2.663147589403113) q[10];
rz(0.12224147248245122) q[10];
ry(1.2881286520926427) q[11];
rz(-1.6854540641488134) q[11];
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
ry(-1.2776982520645157) q[0];
rz(-2.946068340506539) q[0];
ry(-1.5660401590054198) q[1];
rz(1.5696109324203507) q[1];
ry(-1.5810832038153078) q[2];
rz(1.5704497184725625) q[2];
ry(0.0005603527162358902) q[3];
rz(0.9585052450044458) q[3];
ry(-1.5704914563743602) q[4];
rz(3.07807541365812) q[4];
ry(-3.141227546275332) q[5];
rz(-1.8025346441063013) q[5];
ry(-3.1383823985635635) q[6];
rz(-1.5148012790236716) q[6];
ry(0.0016028050023511042) q[7];
rz(2.7073017803232857) q[7];
ry(-1.6914425082291524) q[8];
rz(-1.5733869824620177) q[8];
ry(-1.5807505935708968) q[9];
rz(1.5720622740266021) q[9];
ry(-0.03183254701605431) q[10];
rz(0.4711296144990449) q[10];
ry(1.5277509027564489) q[11];
rz(2.8476992882579846) q[11];
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
ry(-3.1352404347082983) q[0];
rz(-1.8458324514606763) q[0];
ry(1.5712365702674873) q[1];
rz(-2.3077876418236314) q[1];
ry(1.5707708213646159) q[2];
rz(0.45448129333465753) q[2];
ry(1.5706889102528099) q[3];
rz(2.5366906267353495) q[3];
ry(3.140577644125043) q[4];
rz(-2.0288339594701683) q[4];
ry(-1.5714948447017436) q[5];
rz(-2.3228669112322367) q[5];
ry(1.093275754320941) q[6];
rz(2.3963575439894345) q[6];
ry(-1.5693391999806006) q[7];
rz(2.4134928678515113) q[7];
ry(1.5686888648294506) q[8];
rz(2.0626854314942085) q[8];
ry(-1.5700470311938055) q[9];
rz(2.4714205652637435) q[9];
ry(1.570703572733513) q[10];
rz(0.3357815488646807) q[10];
ry(-0.5943100478170891) q[11];
rz(2.9477326896537974) q[11];