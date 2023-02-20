OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(3.1098072935410777) q[0];
rz(-0.05275174687005179) q[0];
ry(-1.5695414474528988) q[1];
rz(-1.572705567870667) q[1];
ry(2.370277882084208) q[2];
rz(2.2297718472676857) q[2];
ry(1.2698903343062193) q[3];
rz(1.6599098413443771) q[3];
ry(-2.286506726860499) q[4];
rz(-2.555043363636938) q[4];
ry(0.16973839215432435) q[5];
rz(2.5416319089722275) q[5];
ry(-0.6575665682266996) q[6];
rz(-2.699421272612129) q[6];
ry(-1.055397654686197) q[7];
rz(2.2152659839596156) q[7];
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
ry(-1.5576642709652404) q[0];
rz(2.7462974259002646) q[0];
ry(0.809497205878186) q[1];
rz(-0.03548119432002128) q[1];
ry(-2.9517017491357893) q[2];
rz(-2.102046235760182) q[2];
ry(1.5697038481681556) q[3];
rz(1.569655783306956) q[3];
ry(-0.42375426602309996) q[4];
rz(-0.5071487779818921) q[4];
ry(1.0053074024391218) q[5];
rz(-2.7052713615478243) q[5];
ry(-0.9036033918429628) q[6];
rz(-0.6863288548024586) q[6];
ry(1.5710711805200654) q[7];
rz(2.5674622124137) q[7];
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
ry(-0.005329343829311384) q[0];
rz(-0.1809119517641777) q[0];
ry(-3.1113510738067935) q[1];
rz(3.1022121446160082) q[1];
ry(1.5756197454449548) q[2];
rz(3.0029467225223763) q[2];
ry(1.5699864618657493) q[3];
rz(-1.1496636568612617) q[3];
ry(1.9460385570311463) q[4];
rz(-0.6451324871926846) q[4];
ry(3.140158189691569) q[5];
rz(0.7399989309539052) q[5];
ry(0.5057539220724523) q[6];
rz(-3.0880076032842805) q[6];
ry(-1.2353616016536764) q[7];
rz(-1.2629674814350516) q[7];
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
ry(-3.1393381667671427) q[0];
rz(-0.5668675625595235) q[0];
ry(1.5729353320897186) q[1];
rz(1.878719426581216) q[1];
ry(0.0007951690787377785) q[2];
rz(-1.4338927884253447) q[2];
ry(1.5725522884418113) q[3];
rz(-0.03678030122697695) q[3];
ry(3.1407053558887545) q[4];
rz(2.3524906633286435) q[4];
ry(-0.0005043217605459077) q[5];
rz(-1.3847619747639475) q[5];
ry(1.942958042137975) q[6];
rz(-2.143712199469226) q[6];
ry(-2.0489599935563634) q[7];
rz(0.6811848156046478) q[7];
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
ry(1.5771251938441173) q[0];
rz(2.7684671387696826) q[0];
ry(-0.28760641775362666) q[1];
rz(-1.764297101455936) q[1];
ry(-1.569558175732611) q[2];
rz(3.0810513901545344) q[2];
ry(-2.927879251500789) q[3];
rz(-0.038178504909911475) q[3];
ry(1.0443866424293033) q[4];
rz(-1.8379973149480793) q[4];
ry(-3.1403134139349445) q[5];
rz(-0.7551047744516799) q[5];
ry(-1.8773127858143812) q[6];
rz(3.107232294781439) q[6];
ry(-1.0416505693183096) q[7];
rz(-1.585816014132874) q[7];
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
ry(0.006787493182621063) q[0];
rz(-2.8042291480275465) q[0];
ry(1.564415791693083) q[1];
rz(-1.8971217196276768) q[1];
ry(-1.690463720141716) q[2];
rz(1.3441289730163481) q[2];
ry(0.7752603020202792) q[3];
rz(1.5758948552596863) q[3];
ry(0.4287029403588445) q[4];
rz(-1.5354634106795768) q[4];
ry(3.139523884371498) q[5];
rz(-1.6250956209411838) q[5];
ry(-2.061935985089563) q[6];
rz(-0.0068118731159245004) q[6];
ry(1.0288123761908585) q[7];
rz(-0.9478146196647712) q[7];
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
ry(-3.1401116874609256) q[0];
rz(-0.3546197419664034) q[0];
ry(2.0279426059866363) q[1];
rz(2.4874017064810166) q[1];
ry(1.569128295820291) q[2];
rz(-1.5630879131912703) q[2];
ry(-1.4767662866523161) q[3];
rz(-1.4682735944932617) q[3];
ry(1.504888278774458) q[4];
rz(2.0288921381836964) q[4];
ry(2.8242670262100593) q[5];
rz(-1.2439828823644472) q[5];
ry(2.4546189776199436) q[6];
rz(2.8707532589396147) q[6];
ry(2.078949606818823) q[7];
rz(1.9471811789744606) q[7];
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
ry(3.1392045925808234) q[0];
rz(1.1983015390631575) q[0];
ry(0.41390958176569903) q[1];
rz(-3.111840546148005) q[1];
ry(1.5722363356390976) q[2];
rz(1.8676127957725253) q[2];
ry(-3.1033898329929537) q[3];
rz(1.7113053421489721) q[3];
ry(-3.138120338029828) q[4];
rz(-2.01993628318126) q[4];
ry(-3.1412134929395794) q[5];
rz(-1.2395409054256135) q[5];
ry(0.8518988324375435) q[6];
rz(2.127218493202341) q[6];
ry(-1.1109087980754535) q[7];
rz(0.9452105179046858) q[7];
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
ry(-1.5765101808504562) q[0];
rz(1.5603072967185518) q[0];
ry(2.5864452844946357) q[1];
rz(-2.095445491164269) q[1];
ry(-2.0163907321108434) q[2];
rz(2.2623155240884527) q[2];
ry(-1.5781689710334492) q[3];
rz(1.3486661809162985) q[3];
ry(0.007052138959278345) q[4];
rz(-2.2606455130601044) q[4];
ry(-1.4240355840253283) q[5];
rz(1.4842205635543217) q[5];
ry(-0.05948504701734334) q[6];
rz(0.6938803920672248) q[6];
ry(-2.8659283867947556) q[7];
rz(2.6626269390186623) q[7];
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
ry(-1.4990072945458854) q[0];
rz(-1.4836285587760631) q[0];
ry(-0.0011454073206325077) q[1];
rz(0.07155838491285937) q[1];
ry(-0.011156557439113257) q[2];
rz(2.9938720393786484) q[2];
ry(1.570837801052391) q[3];
rz(-3.1405748622094145) q[3];
ry(0.0018332977178855197) q[4];
rz(0.11100178516122926) q[4];
ry(0.1395033807421937) q[5];
rz(1.6590740137659052) q[5];
ry(-0.8172916515104891) q[6];
rz(2.6526073748375576) q[6];
ry(1.5266680868649356) q[7];
rz(1.04169815449826) q[7];
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
ry(1.6294757573621637) q[0];
rz(-2.2649905627975455) q[0];
ry(1.571547323613676) q[1];
rz(-0.007292827709729721) q[1];
ry(-0.0010435437912414969) q[2];
rz(2.8137605496203824) q[2];
ry(1.561083354595545) q[3];
rz(-0.0023990928565753578) q[3];
ry(-0.03957322455574092) q[4];
rz(3.016260745476123) q[4];
ry(1.5692077900633223) q[5];
rz(2.6544311501131506) q[5];
ry(3.1245216965455658) q[6];
rz(-0.6432077929080258) q[6];
ry(-1.1587673608342304) q[7];
rz(-0.38507899323708433) q[7];
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
ry(0.10253928760548181) q[0];
rz(1.6797517835003173) q[0];
ry(2.9779336228824675) q[1];
rz(-0.0030761043397698608) q[1];
ry(-0.00016527236840406626) q[2];
rz(-1.8668661557510127) q[2];
ry(-1.5774023242703903) q[3];
rz(3.037483131234036) q[3];
ry(-1.7801701695968912) q[4];
rz(-2.5264593506811255) q[4];
ry(3.134470357846059) q[5];
rz(-0.4840851871392217) q[5];
ry(0.05790209821753828) q[6];
rz(-0.7808550506261713) q[6];
ry(3.140636103970918) q[7];
rz(1.9956377264689857) q[7];
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
ry(-1.5801644903625673) q[0];
rz(-2.147475870202648) q[0];
ry(2.935039442516771) q[1];
rz(1.5800325074160009) q[1];
ry(1.5690550093888973) q[2];
rz(-1.5712919507957004) q[2];
ry(0.025573733679520837) q[3];
rz(-3.094114678176178) q[3];
ry(1.5418687454116746) q[4];
rz(-1.6000956015350765) q[4];
ry(-1.569159555050324) q[5];
rz(1.6253069637060857) q[5];
ry(1.5664525980846877) q[6];
rz(1.9521895375927962) q[6];
ry(-2.432487080471151) q[7];
rz(0.3055772747600267) q[7];
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
ry(-3.140692879366505) q[0];
rz(3.012625119352369) q[0];
ry(-0.08004392310337691) q[1];
rz(2.961357660766863) q[1];
ry(1.5716191190928237) q[2];
rz(2.1185479589603946) q[2];
ry(1.5653700529492411) q[3];
rz(-0.18300516806197287) q[3];
ry(1.570337685318088) q[4];
rz(2.1267647352361143) q[4];
ry(-1.5704148546475087) q[5];
rz(2.957689710257437) q[5];
ry(-0.0003742714491929566) q[6];
rz(-2.975276182269804) q[6];
ry(2.4691630275641194) q[7];
rz(2.9663043194117895) q[7];