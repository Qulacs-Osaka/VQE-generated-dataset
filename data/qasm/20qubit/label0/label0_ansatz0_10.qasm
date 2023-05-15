OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[0],q[1];
rz(-0.08520078506504608) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0676801865751068) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04871743165649483) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.06780565055008034) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.024707575340957175) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.09919554127867093) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.06764582586839114) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.018214403582603875) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.014498313481752059) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.07679578765293406) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.0328540257133975) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.0021645346809696917) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.08907156353233057) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.003989535269204731) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.06329594938207836) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.08833557312347882) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.018491940882915857) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.08403689112864587) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.09142670400254717) q[19];
cx q[18],q[19];
h q[0];
rz(0.6201112508105577) q[0];
h q[0];
h q[1];
rz(0.5932974038834735) q[1];
h q[1];
h q[2];
rz(1.0674908363195479) q[2];
h q[2];
h q[3];
rz(0.47046349855933356) q[3];
h q[3];
h q[4];
rz(0.10465696636969066) q[4];
h q[4];
h q[5];
rz(0.8017111985864296) q[5];
h q[5];
h q[6];
rz(0.7978395011359208) q[6];
h q[6];
h q[7];
rz(-0.0626188285995932) q[7];
h q[7];
h q[8];
rz(0.14726178116754213) q[8];
h q[8];
h q[9];
rz(0.6843236291782672) q[9];
h q[9];
h q[10];
rz(0.33386595147310716) q[10];
h q[10];
h q[11];
rz(0.7372673409268503) q[11];
h q[11];
h q[12];
rz(0.7390410190264934) q[12];
h q[12];
h q[13];
rz(1.0493056378364172) q[13];
h q[13];
h q[14];
rz(0.5336405525770161) q[14];
h q[14];
h q[15];
rz(0.6663820730629991) q[15];
h q[15];
h q[16];
rz(0.8189045251383643) q[16];
h q[16];
h q[17];
rz(0.5821092527569237) q[17];
h q[17];
h q[18];
rz(0.2745018968613262) q[18];
h q[18];
h q[19];
rz(0.5000044217442576) q[19];
h q[19];
rz(-0.4809405950434381) q[0];
rz(0.17244687826472688) q[1];
rz(0.11727826325084562) q[2];
rz(-0.2068224335659552) q[3];
rz(-0.4217845897224436) q[4];
rz(0.39052839616451523) q[5];
rz(-0.1463203768827889) q[6];
rz(0.34240443341196836) q[7];
rz(-0.15926789039085223) q[8];
rz(-0.38797384797242396) q[9];
rz(-0.16851505804890826) q[10];
rz(-0.2563804073806418) q[11];
rz(0.03679158393998631) q[12];
rz(-0.6234553956508064) q[13];
rz(-0.1498379008587066) q[14];
rz(-0.14483164643737467) q[15];
rz(-0.15101875295888484) q[16];
rz(-0.3902268133219697) q[17];
rz(0.11927919741774275) q[18];
rz(-0.22654375379643923) q[19];
cx q[0],q[1];
rz(-0.2850696717816037) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.06569167638716769) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.013261750403522177) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.11080919077202753) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.49053746945107396) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.7247511904914264) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.12552238649580194) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.2519272162872488) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.3683067388316033) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.44821673240277116) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.36488287395906166) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.01052885887252248) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.5246056362916393) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.01721435599646285) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(0.0709623199076125) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.04182676208878332) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.5967025278825812) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(0.02468747509310439) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.09653633365629996) q[19];
cx q[18],q[19];
h q[0];
rz(0.2237653602904834) q[0];
h q[0];
h q[1];
rz(0.6457841073890584) q[1];
h q[1];
h q[2];
rz(0.6171865404924501) q[2];
h q[2];
h q[3];
rz(0.49189054598451165) q[3];
h q[3];
h q[4];
rz(0.21340494187007958) q[4];
h q[4];
h q[5];
rz(-0.13349481623272708) q[5];
h q[5];
h q[6];
rz(0.48371329098347776) q[6];
h q[6];
h q[7];
rz(0.14714002950302968) q[7];
h q[7];
h q[8];
rz(0.014548532222549114) q[8];
h q[8];
h q[9];
rz(0.5156760861562818) q[9];
h q[9];
h q[10];
rz(0.23901234026461665) q[10];
h q[10];
h q[11];
rz(0.5393325166781247) q[11];
h q[11];
h q[12];
rz(0.6132375035732546) q[12];
h q[12];
h q[13];
rz(0.7148356987020358) q[13];
h q[13];
h q[14];
rz(0.6947969424195873) q[14];
h q[14];
h q[15];
rz(0.6092026093736811) q[15];
h q[15];
h q[16];
rz(0.049354102266424796) q[16];
h q[16];
h q[17];
rz(0.3536775738157325) q[17];
h q[17];
h q[18];
rz(0.8264503352500281) q[18];
h q[18];
h q[19];
rz(0.56812445559979) q[19];
h q[19];
rz(-0.6363486837925868) q[0];
rz(0.4254585625425347) q[1];
rz(0.2655120503928983) q[2];
rz(-0.30342499485561886) q[3];
rz(-0.4984698813148581) q[4];
rz(0.472081776110313) q[5];
rz(-0.7560745818629593) q[6];
rz(0.22302026828155871) q[7];
rz(-0.32434832503203703) q[8];
rz(-0.47440256419948457) q[9];
rz(-0.05072824382678551) q[10];
rz(-0.5261013194116765) q[11];
rz(-0.3210345524372653) q[12];
rz(-0.6372393953223585) q[13];
rz(-0.24799977786413893) q[14];
rz(-0.39576550446177094) q[15];
rz(-0.2642467483281736) q[16];
rz(-0.6260508013621786) q[17];
rz(0.10274525846762791) q[18];
rz(-0.29230135199146906) q[19];
cx q[0],q[1];
rz(-0.0631988890909729) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.21394778999856304) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.052939019092874295) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.06437435283553333) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.4723259817111219) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.8531004507526279) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.21769024473822604) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.02868999479245219) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.3439505263349843) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.06228236961827437) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.019911611171531618) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.02287962094953524) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.47962821667341965) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(0.12665443976901186) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.09060038626620294) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.04571476833180218) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.7901260362810834) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(0.1274005427909374) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.43169474474220776) q[19];
cx q[18],q[19];
h q[0];
rz(0.03667688104272113) q[0];
h q[0];
h q[1];
rz(0.659097255206465) q[1];
h q[1];
h q[2];
rz(0.005894627583416282) q[2];
h q[2];
h q[3];
rz(0.3882741055176317) q[3];
h q[3];
h q[4];
rz(0.7339148159324558) q[4];
h q[4];
h q[5];
rz(0.48300950039249524) q[5];
h q[5];
h q[6];
rz(-0.9113772906786681) q[6];
h q[6];
h q[7];
rz(0.1638855578041591) q[7];
h q[7];
h q[8];
rz(0.8419374814305508) q[8];
h q[8];
h q[9];
rz(0.46042949135907496) q[9];
h q[9];
h q[10];
rz(0.4430546633679608) q[10];
h q[10];
h q[11];
rz(0.4759042189582279) q[11];
h q[11];
h q[12];
rz(0.6089027418647197) q[12];
h q[12];
h q[13];
rz(0.6927041810875594) q[13];
h q[13];
h q[14];
rz(0.42258177127210106) q[14];
h q[14];
h q[15];
rz(0.3214260166791861) q[15];
h q[15];
h q[16];
rz(0.815843531386669) q[16];
h q[16];
h q[17];
rz(0.9353640260729187) q[17];
h q[17];
h q[18];
rz(0.2266880500556301) q[18];
h q[18];
h q[19];
rz(0.4276120351101834) q[19];
h q[19];
rz(-0.49994544014306047) q[0];
rz(0.30303801245088985) q[1];
rz(0.13246030983753806) q[2];
rz(-0.4370101574932805) q[3];
rz(0.18132567925270346) q[4];
rz(0.14594824832583797) q[5];
rz(-0.020956712921607324) q[6];
rz(0.24343890436879356) q[7];
rz(-0.5105607928284828) q[8];
rz(-0.5101812827394134) q[9];
rz(0.31859302051110994) q[10];
rz(-0.5977694179676302) q[11];
rz(-0.5623754804461378) q[12];
rz(0.07171774343971715) q[13];
rz(-0.28213097486524813) q[14];
rz(-0.5724899170249964) q[15];
rz(-0.18036745429904913) q[16];
rz(-0.24045349016416928) q[17];
rz(-0.040813768191847254) q[18];
rz(-0.11351889753269873) q[19];
cx q[0],q[1];
rz(0.21050818682557257) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.3455656172957024) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.26488622848918714) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.04786950118601851) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.24926373197982846) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.37493341540124575) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.3512948142127447) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.12412961408863661) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.025146128535394285) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.29849598537235345) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.0646137661559343) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.12061452146188348) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(0.1442179079993489) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.0319365894636235) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.025644305679378768) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.029200090435295638) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(0.22490341399886732) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.09215011920295482) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.528470578195593) q[19];
cx q[18],q[19];
h q[0];
rz(-0.007614678792135326) q[0];
h q[0];
h q[1];
rz(0.2543227034257948) q[1];
h q[1];
h q[2];
rz(-0.2786666881165494) q[2];
h q[2];
h q[3];
rz(0.30634512020432747) q[3];
h q[3];
h q[4];
rz(0.6510645194657998) q[4];
h q[4];
h q[5];
rz(0.5290478249225481) q[5];
h q[5];
h q[6];
rz(-0.49339735388721867) q[6];
h q[6];
h q[7];
rz(0.48818974448214936) q[7];
h q[7];
h q[8];
rz(0.6173854589006006) q[8];
h q[8];
h q[9];
rz(0.33431691177286316) q[9];
h q[9];
h q[10];
rz(0.6784764789176307) q[10];
h q[10];
h q[11];
rz(0.8769652485329393) q[11];
h q[11];
h q[12];
rz(-0.23081280388325942) q[12];
h q[12];
h q[13];
rz(1.1128988146858976) q[13];
h q[13];
h q[14];
rz(-0.07733868757059255) q[14];
h q[14];
h q[15];
rz(-0.007060264772186941) q[15];
h q[15];
h q[16];
rz(0.8905535921088403) q[16];
h q[16];
h q[17];
rz(0.852903674292174) q[17];
h q[17];
h q[18];
rz(0.23484973115416352) q[18];
h q[18];
h q[19];
rz(0.5215435086376634) q[19];
h q[19];
rz(-0.31939112319771734) q[0];
rz(0.0739389754689527) q[1];
rz(0.3914456008780604) q[2];
rz(-0.3698104746028676) q[3];
rz(0.9635421793171013) q[4];
rz(0.19120598831188462) q[5];
rz(-0.3501052136472478) q[6];
rz(0.5076784314328061) q[7];
rz(-0.45318098937387313) q[8];
rz(-0.3437724454830037) q[9];
rz(0.4742776466618066) q[10];
rz(-0.08012292858256653) q[11];
rz(-0.5208495302364938) q[12];
rz(-0.15286116203929592) q[13];
rz(-0.44066963276757315) q[14];
rz(-0.49983628055928414) q[15];
rz(-0.21181325209384685) q[16];
rz(-0.637920501980858) q[17];
rz(0.025015946479204024) q[18];
rz(-0.13329965494988666) q[19];
cx q[0],q[1];
rz(0.05763162130050947) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.23559240007972695) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.18036319354131905) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.04446585421185177) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.4776351722798053) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.0511728485023643) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.11170477415342916) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.20445853769619568) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.1560581866455782) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(1.0502493693748272) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.2919161832369467) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.2028915880381449) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(0.5630655471307556) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.24370882111744177) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.8571000214326209) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.3262634401445867) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(0.7870376677572581) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.03982570409346747) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.45539646002073153) q[19];
cx q[18],q[19];
h q[0];
rz(-0.007245937837242769) q[0];
h q[0];
h q[1];
rz(0.031252093541991204) q[1];
h q[1];
h q[2];
rz(0.03749206471671371) q[2];
h q[2];
h q[3];
rz(-0.05306841546906201) q[3];
h q[3];
h q[4];
rz(-0.0602925167020187) q[4];
h q[4];
h q[5];
rz(0.3759263691748617) q[5];
h q[5];
h q[6];
rz(-0.020085684511870976) q[6];
h q[6];
h q[7];
rz(0.7690668596937732) q[7];
h q[7];
h q[8];
rz(0.11927181764934897) q[8];
h q[8];
h q[9];
rz(0.34414872754586523) q[9];
h q[9];
h q[10];
rz(0.06145553478832923) q[10];
h q[10];
h q[11];
rz(0.7638017774469033) q[11];
h q[11];
h q[12];
rz(0.40668976990403644) q[12];
h q[12];
h q[13];
rz(0.7176162568356393) q[13];
h q[13];
h q[14];
rz(-0.06497473272938999) q[14];
h q[14];
h q[15];
rz(0.7900456791201339) q[15];
h q[15];
h q[16];
rz(-0.002536133958925489) q[16];
h q[16];
h q[17];
rz(0.7422008998556968) q[17];
h q[17];
h q[18];
rz(0.358692388616837) q[18];
h q[18];
h q[19];
rz(0.533588911714158) q[19];
h q[19];
rz(-0.1881896406523975) q[0];
rz(-0.15862293460043048) q[1];
rz(0.2488156859207296) q[2];
rz(-0.21418489098487611) q[3];
rz(0.5224350257618419) q[4];
rz(-0.021062138846119273) q[5];
rz(-0.280496289970714) q[6];
rz(0.39957075190981606) q[7];
rz(-0.5358917816872146) q[8];
rz(-0.14410893928611646) q[9];
rz(0.31516430404049917) q[10];
rz(0.004994584649732817) q[11];
rz(-0.07985034286194401) q[12];
rz(-0.01904089543027843) q[13];
rz(-0.32471305071167866) q[14];
rz(-0.034991420195587264) q[15];
rz(-0.4790844661305679) q[16];
rz(-0.13067282565239133) q[17];
rz(-0.0092419432293757) q[18];
rz(-0.25709423174881896) q[19];
cx q[0],q[1];
rz(-0.15499269236575536) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.23653518708078083) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.16306259959855363) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.06841620258771301) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.8231403587188405) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.15997246046803457) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.2115307846521923) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.0887033828544411) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.8064000800030926) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.4831778277490426) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.039273850127119786) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.27346905448228986) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.9496521670551649) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(0.0045461835629543555) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(0.015209811742964601) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.07944725262983728) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.47303726351522196) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(0.17939489184161017) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.2138887345828543) q[19];
cx q[18],q[19];
h q[0];
rz(-0.0096161873331535) q[0];
h q[0];
h q[1];
rz(-0.13078517449980445) q[1];
h q[1];
h q[2];
rz(0.31172440165479215) q[2];
h q[2];
h q[3];
rz(0.1631985194051096) q[3];
h q[3];
h q[4];
rz(-0.28259414632968854) q[4];
h q[4];
h q[5];
rz(-0.26695026638761155) q[5];
h q[5];
h q[6];
rz(0.06806163508262542) q[6];
h q[6];
h q[7];
rz(0.0165217133887792) q[7];
h q[7];
h q[8];
rz(0.20027567563014242) q[8];
h q[8];
h q[9];
rz(1.7510621238912718) q[9];
h q[9];
h q[10];
rz(-0.16681172395591204) q[10];
h q[10];
h q[11];
rz(0.9273151703560899) q[11];
h q[11];
h q[12];
rz(-0.256062462184527) q[12];
h q[12];
h q[13];
rz(0.5531169289122388) q[13];
h q[13];
h q[14];
rz(0.02317760672332756) q[14];
h q[14];
h q[15];
rz(0.6612893052841379) q[15];
h q[15];
h q[16];
rz(-0.05048254708704715) q[16];
h q[16];
h q[17];
rz(0.357219547213925) q[17];
h q[17];
h q[18];
rz(0.4435210987723509) q[18];
h q[18];
h q[19];
rz(0.2761042667783876) q[19];
h q[19];
rz(-0.016391381333822674) q[0];
rz(-0.0741776308975673) q[1];
rz(0.05164321358715695) q[2];
rz(-0.2067331494727093) q[3];
rz(-0.0373180011589952) q[4];
rz(-0.20836218008597965) q[5];
rz(-0.27398862919036665) q[6];
rz(0.38076468719079687) q[7];
rz(-0.04207715908185621) q[8];
rz(-0.12157150911937506) q[9];
rz(0.6159863352690601) q[10];
rz(0.012567695339199365) q[11];
rz(0.017563339050159815) q[12];
rz(0.1363485302383654) q[13];
rz(-0.14976000515792032) q[14];
rz(0.07820132779276129) q[15];
rz(-0.3998201085819376) q[16];
rz(-0.08402993103191571) q[17];
rz(-0.012577986373980353) q[18];
rz(-0.25240083669692753) q[19];
cx q[0],q[1];
rz(-0.07002842117627192) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.11295569794469769) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.12940541029555563) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.3002418623115708) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(1.003025323515213) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(1.0054298844539515) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.3563101203279877) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.02118537242332333) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.6634971204143197) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.5255809773267458) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.2426618506653307) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.5516320345388558) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.1573603406073407) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(0.19622758494398712) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(0.6187369268937406) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.7369716835594143) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(0.04318607192150104) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.12192300719872078) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.39143266570976215) q[19];
cx q[18],q[19];
h q[0];
rz(0.006344013319096587) q[0];
h q[0];
h q[1];
rz(-0.1887954698822467) q[1];
h q[1];
h q[2];
rz(0.11970160487680369) q[2];
h q[2];
h q[3];
rz(-0.30954626114100803) q[3];
h q[3];
h q[4];
rz(0.4175052855189307) q[4];
h q[4];
h q[5];
rz(-0.41436328980770865) q[5];
h q[5];
h q[6];
rz(0.5629488155994521) q[6];
h q[6];
h q[7];
rz(0.2755319629132091) q[7];
h q[7];
h q[8];
rz(-0.8391727029948722) q[8];
h q[8];
h q[9];
rz(0.3076757153417703) q[9];
h q[9];
h q[10];
rz(-0.3179677217788772) q[10];
h q[10];
h q[11];
rz(1.1390727083274033) q[11];
h q[11];
h q[12];
rz(0.21524070175108573) q[12];
h q[12];
h q[13];
rz(0.23279955159661608) q[13];
h q[13];
h q[14];
rz(-0.18298577944295483) q[14];
h q[14];
h q[15];
rz(0.07846198499538058) q[15];
h q[15];
h q[16];
rz(-0.1830183591381179) q[16];
h q[16];
h q[17];
rz(0.0441495675087703) q[17];
h q[17];
h q[18];
rz(0.3340359885234463) q[18];
h q[18];
h q[19];
rz(0.16692391383403607) q[19];
h q[19];
rz(0.0113056584080656) q[0];
rz(0.2717050789504758) q[1];
rz(0.16396622165299718) q[2];
rz(-0.046852260704364004) q[3];
rz(0.046886096359925786) q[4];
rz(0.2783974035058131) q[5];
rz(0.04940770715833923) q[6];
rz(0.17594304917729106) q[7];
rz(-0.025840654039764946) q[8];
rz(-0.3045443491240601) q[9];
rz(0.20339224721370494) q[10];
rz(-0.013071378342858287) q[11];
rz(0.01407875453197492) q[12];
rz(-0.13450534743854436) q[13];
rz(-0.057556062062108684) q[14];
rz(-0.063008941461342) q[15];
rz(-0.20373877615306776) q[16];
rz(-0.04745601607177292) q[17];
rz(-0.012258255800873122) q[18];
rz(-0.14320318236121468) q[19];
cx q[0],q[1];
rz(0.2733872605962458) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.2724015600523417) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.017665078767257293) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.050237121932102065) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.4128845595911326) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.1007322259309771) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.6111979882222675) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.37241078348497536) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.46029587470872774) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.23492723237505228) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.3197659726995227) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.5618508757132457) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(0.5559821031790865) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.19224968905068546) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.3751125182844055) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.25207313060160214) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.5931443701951702) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.04394867790444006) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.29755523862801597) q[19];
cx q[18],q[19];
h q[0];
rz(0.5418895243103253) q[0];
h q[0];
h q[1];
rz(-0.410206493289441) q[1];
h q[1];
h q[2];
rz(0.2782545811407852) q[2];
h q[2];
h q[3];
rz(-0.5400115702326161) q[3];
h q[3];
h q[4];
rz(0.3479476855695516) q[4];
h q[4];
h q[5];
rz(-0.22621459428673435) q[5];
h q[5];
h q[6];
rz(-0.34600898432179467) q[6];
h q[6];
h q[7];
rz(0.045996443378746824) q[7];
h q[7];
h q[8];
rz(-0.15550718845905187) q[8];
h q[8];
h q[9];
rz(0.22173916186177334) q[9];
h q[9];
h q[10];
rz(-0.06233974935321225) q[10];
h q[10];
h q[11];
rz(0.6770650543914332) q[11];
h q[11];
h q[12];
rz(-0.17649000347978466) q[12];
h q[12];
h q[13];
rz(-0.3575436541551493) q[13];
h q[13];
h q[14];
rz(-0.3899230540153679) q[14];
h q[14];
h q[15];
rz(-0.019831149015185838) q[15];
h q[15];
h q[16];
rz(0.10511178782328297) q[16];
h q[16];
h q[17];
rz(0.21835368565118818) q[17];
h q[17];
h q[18];
rz(0.42254405920896837) q[18];
h q[18];
h q[19];
rz(0.2626441805508392) q[19];
h q[19];
rz(0.11750203399419143) q[0];
rz(0.02924589742823515) q[1];
rz(0.10026914408992045) q[2];
rz(0.051406731025849896) q[3];
rz(0.011194986476253985) q[4];
rz(-0.12475841451251164) q[5];
rz(-0.26080350633939386) q[6];
rz(0.5615773074033353) q[7];
rz(-0.006290993774155553) q[8];
rz(0.11004523502005366) q[9];
rz(0.7817345357520342) q[10];
rz(0.061564792452365305) q[11];
rz(0.007442056167219688) q[12];
rz(0.04397714843923564) q[13];
rz(0.04307898648864188) q[14];
rz(0.015668826613815797) q[15];
rz(0.26207581978485683) q[16];
rz(0.01211516716435552) q[17];
rz(0.032152166293088) q[18];
rz(-0.26390876801819885) q[19];
cx q[0],q[1];
rz(0.03607275917458083) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.26132012868998244) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.7106728285726859) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.08566285240728298) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.6007932450705787) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.44717331143538935) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.8967622556847759) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.3707257878573191) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.41278522152604497) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.14315602710190137) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.5548445142043381) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.6650632794171539) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.181102349521537) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(0.1827146845312533) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.12112624032848596) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.049821580671891984) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.5168211795112094) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.04023960950655024) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.46503284213018986) q[19];
cx q[18],q[19];
h q[0];
rz(0.5459200876538104) q[0];
h q[0];
h q[1];
rz(-0.2622649528364482) q[1];
h q[1];
h q[2];
rz(-0.5176104804333499) q[2];
h q[2];
h q[3];
rz(-0.16055805245069765) q[3];
h q[3];
h q[4];
rz(-0.5016826206725261) q[4];
h q[4];
h q[5];
rz(-1.1825200417708663) q[5];
h q[5];
h q[6];
rz(0.045202564158853646) q[6];
h q[6];
h q[7];
rz(-0.13321187303242657) q[7];
h q[7];
h q[8];
rz(-1.3089528697298034) q[8];
h q[8];
h q[9];
rz(-0.19253444400721162) q[9];
h q[9];
h q[10];
rz(0.05526671489031023) q[10];
h q[10];
h q[11];
rz(0.2877721417624368) q[11];
h q[11];
h q[12];
rz(-0.34237474569335286) q[12];
h q[12];
h q[13];
rz(-0.43990187328305463) q[13];
h q[13];
h q[14];
rz(-0.6231100504537176) q[14];
h q[14];
h q[15];
rz(0.4259543806401826) q[15];
h q[15];
h q[16];
rz(-0.26709237012429304) q[16];
h q[16];
h q[17];
rz(0.8237054072346636) q[17];
h q[17];
h q[18];
rz(0.4532368133013328) q[18];
h q[18];
h q[19];
rz(0.07429974499388224) q[19];
h q[19];
rz(0.15722889528917025) q[0];
rz(0.12046863888245904) q[1];
rz(-0.32487918034722074) q[2];
rz(0.05125276427173114) q[3];
rz(-0.022273806176109617) q[4];
rz(-0.041630845422672896) q[5];
rz(-0.14542699502159281) q[6];
rz(-0.4269828927108146) q[7];
rz(0.027018703007092013) q[8];
rz(0.31444227998377433) q[9];
rz(1.2464653753637334) q[10];
rz(-0.09612994910441251) q[11];
rz(-0.011864423460907221) q[12];
rz(-0.04561935007413682) q[13];
rz(-0.0016824504379527457) q[14];
rz(0.09790331862712945) q[15];
rz(0.0718586546656794) q[16];
rz(0.01779220478521228) q[17];
rz(0.005150256370526015) q[18];
rz(-0.23187240648952598) q[19];
cx q[0],q[1];
rz(0.5204514416096724) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.5724351970596464) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.6842585883395234) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.1032915448804337) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.02964066845964617) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.7463595970621716) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(1.0829488462074521) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.8834275678785547) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.7724225019624615) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.2999519238162872) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.250627261101974) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.15254674264957144) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.4230066015037017) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.03652490416730328) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(0.30852626510619996) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.36363601793106565) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(0.5962856988925375) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.7745732809037025) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.34554004022150314) q[19];
cx q[18],q[19];
h q[0];
rz(0.18262184622624464) q[0];
h q[0];
h q[1];
rz(1.0413556337987344) q[1];
h q[1];
h q[2];
rz(-0.6192322975117804) q[2];
h q[2];
h q[3];
rz(-0.05294529765973095) q[3];
h q[3];
h q[4];
rz(-0.3939881719263417) q[4];
h q[4];
h q[5];
rz(-1.5796864289906187) q[5];
h q[5];
h q[6];
rz(1.5157262242373661) q[6];
h q[6];
h q[7];
rz(-0.9137079061165372) q[7];
h q[7];
h q[8];
rz(0.06420185049540912) q[8];
h q[8];
h q[9];
rz(-1.3088438074115882) q[9];
h q[9];
h q[10];
rz(-0.001963569359600989) q[10];
h q[10];
h q[11];
rz(0.46691149726250814) q[11];
h q[11];
h q[12];
rz(-0.22400566495519478) q[12];
h q[12];
h q[13];
rz(-0.5992440558323893) q[13];
h q[13];
h q[14];
rz(-0.8073485011839306) q[14];
h q[14];
h q[15];
rz(-0.061371155918367754) q[15];
h q[15];
h q[16];
rz(0.4467512286567433) q[16];
h q[16];
h q[17];
rz(-0.23435649403517494) q[17];
h q[17];
h q[18];
rz(0.31569504881925214) q[18];
h q[18];
h q[19];
rz(-0.009901357146355172) q[19];
h q[19];
rz(0.17112253852084716) q[0];
rz(-0.41171982664116796) q[1];
rz(0.015352519347638201) q[2];
rz(-0.018475066046870228) q[3];
rz(0.36206594371610623) q[4];
rz(-0.022038822680790958) q[5];
rz(0.8765917716735836) q[6];
rz(0.0810128320625958) q[7];
rz(-0.011585825168151799) q[8];
rz(-0.016320263384709686) q[9];
rz(1.0583305891426484) q[10];
rz(0.041539433992731224) q[11];
rz(-0.004079506457627509) q[12];
rz(-0.021234866568729693) q[13];
rz(-0.010912184881951755) q[14];
rz(-0.10645006084880201) q[15];
rz(-0.0027277027762185875) q[16];
rz(-0.0235880265176749) q[17];
rz(-0.00799673808326368) q[18];
rz(-0.2450260020899533) q[19];
cx q[0],q[1];
rz(0.9910961846942753) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.028127733851729946) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.23554020821847668) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.0033767151304361443) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.21676291920454038) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.008967241171338665) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.08531205244956096) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.5477513578420508) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.2758814419023487) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.998986293688222) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.3227842697461683) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.1906632827994206) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.6124208511605014) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.04472405501074775) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(0.022486846024115697) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-1.0901205446896405) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(0.4180175547286358) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.9328074471426225) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.17358948192094137) q[19];
cx q[18],q[19];
h q[0];
rz(0.517913041062249) q[0];
h q[0];
h q[1];
rz(0.4669717788404079) q[1];
h q[1];
h q[2];
rz(-1.5105326693711663) q[2];
h q[2];
h q[3];
rz(0.06929852607653161) q[3];
h q[3];
h q[4];
rz(-0.9378292435374256) q[4];
h q[4];
h q[5];
rz(-0.95741254617589) q[5];
h q[5];
h q[6];
rz(-1.3672770096848743) q[6];
h q[6];
h q[7];
rz(-1.5064502397964625) q[7];
h q[7];
h q[8];
rz(0.23187653641283637) q[8];
h q[8];
h q[9];
rz(0.5548991514452027) q[9];
h q[9];
h q[10];
rz(-1.6457273844866664) q[10];
h q[10];
h q[11];
rz(1.1449396583736364) q[11];
h q[11];
h q[12];
rz(-0.45561794294600877) q[12];
h q[12];
h q[13];
rz(0.6453212532550455) q[13];
h q[13];
h q[14];
rz(-0.9468388108809908) q[14];
h q[14];
h q[15];
rz(0.5195720496394559) q[15];
h q[15];
h q[16];
rz(-0.7342172068207371) q[16];
h q[16];
h q[17];
rz(0.5968936092098237) q[17];
h q[17];
h q[18];
rz(-0.5047857266335373) q[18];
h q[18];
h q[19];
rz(-0.17100959683638936) q[19];
h q[19];
rz(0.713610434559706) q[0];
rz(-0.011875837339922347) q[1];
rz(0.8013072694523873) q[2];
rz(-0.05500501521361226) q[3];
rz(-0.2735830886951257) q[4];
rz(-0.05425458796619243) q[5];
rz(2.7074225264279383) q[6];
rz(0.027703486150690362) q[7];
rz(-0.022433489085409215) q[8];
rz(0.061357969933169824) q[9];
rz(-0.1273924736050001) q[10];
rz(-0.045794424177696774) q[11];
rz(-0.08039055199206067) q[12];
rz(0.014169690613968907) q[13];
rz(0.04484235390722954) q[14];
rz(0.019950141128498085) q[15];
rz(-0.06840398971511702) q[16];
rz(-0.00356267811281167) q[17];
rz(-0.008710084948442522) q[18];
rz(-0.23588532777093715) q[19];
cx q[0],q[1];
rz(0.8458404184993199) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.018350449096967737) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.25846627268747835) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.06253206230986463) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.054087927834121016) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.9484674280415266) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.3637481998843543) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.6533639764395067) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(1.222129500805938) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.34757865980359387) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.13794688331919178) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.5068716881303084) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(0.8981166496271916) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.07848917824718181) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(0.4757183780001254) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.24854360093109765) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.21537348403949103) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(0.5970939368857209) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.5468320217418782) q[19];
cx q[18],q[19];
h q[0];
rz(-0.030994693575761012) q[0];
h q[0];
h q[1];
rz(-0.37201071744375075) q[1];
h q[1];
h q[2];
rz(-0.47812779643941555) q[2];
h q[2];
h q[3];
rz(-0.6171246712026518) q[3];
h q[3];
h q[4];
rz(-1.161865589992222) q[4];
h q[4];
h q[5];
rz(0.5339761855090925) q[5];
h q[5];
h q[6];
rz(-0.19349188114452778) q[6];
h q[6];
h q[7];
rz(-1.2574653907725089) q[7];
h q[7];
h q[8];
rz(-1.0638054097791585) q[8];
h q[8];
h q[9];
rz(-0.2981028492194161) q[9];
h q[9];
h q[10];
rz(-0.9046468358287506) q[10];
h q[10];
h q[11];
rz(0.7629877321076588) q[11];
h q[11];
h q[12];
rz(0.2834268135230503) q[12];
h q[12];
h q[13];
rz(1.1538077985442867) q[13];
h q[13];
h q[14];
rz(-0.8747713103386695) q[14];
h q[14];
h q[15];
rz(-0.8293706022736991) q[15];
h q[15];
h q[16];
rz(-1.283091240505786) q[16];
h q[16];
h q[17];
rz(0.08513804977063495) q[17];
h q[17];
h q[18];
rz(-0.3934663667006595) q[18];
h q[18];
h q[19];
rz(-0.875675834057121) q[19];
h q[19];
rz(0.9899839669466841) q[0];
rz(0.43604949132035065) q[1];
rz(1.2903818484873588) q[2];
rz(-0.027015683419224265) q[3];
rz(0.3020372616612201) q[4];
rz(0.2630105371438321) q[5];
rz(-0.4360951582269885) q[6];
rz(-0.4095248948521421) q[7];
rz(-0.012485302586735612) q[8];
rz(-0.005158903008130332) q[9];
rz(0.1677669944799526) q[10];
rz(0.050282328788743516) q[11];
rz(0.014638901823625422) q[12];
rz(-0.028587912417547892) q[13];
rz(-0.0461457470828598) q[14];
rz(-0.019329487312090474) q[15];
rz(0.026301923982738784) q[16];
rz(0.0023768318111565814) q[17];
rz(-0.0013372452130912386) q[18];
rz(-0.05567539905582164) q[19];
cx q[0],q[1];
rz(1.016749681031679) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.17385421410341204) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.15180833062391877) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.1771948639954051) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.32814183380850037) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.9550905355201419) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.05294938063690141) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.32074000295224386) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.879489652694589) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.40599063633983) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.263536369925831) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.06310952987056359) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(0.15114856182974004) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(0.23694407169062492) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.19277679470774048) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.011652620538741552) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.12254526400836653) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(0.9449750763072264) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.24439806173628342) q[19];
cx q[18],q[19];
h q[0];
rz(-1.2358555168344065) q[0];
h q[0];
h q[1];
rz(-0.5415899114086288) q[1];
h q[1];
h q[2];
rz(0.20330875190971345) q[2];
h q[2];
h q[3];
rz(0.4052346313676708) q[3];
h q[3];
h q[4];
rz(-1.2221252165280658) q[4];
h q[4];
h q[5];
rz(-0.1850042783424289) q[5];
h q[5];
h q[6];
rz(0.055387228844904476) q[6];
h q[6];
h q[7];
rz(-0.03768288469779929) q[7];
h q[7];
h q[8];
rz(-1.2326786414948554) q[8];
h q[8];
h q[9];
rz(-0.15037930714555267) q[9];
h q[9];
h q[10];
rz(1.514738099057597) q[10];
h q[10];
h q[11];
rz(0.2679985946075076) q[11];
h q[11];
h q[12];
rz(-0.23990573344298768) q[12];
h q[12];
h q[13];
rz(0.750616773970109) q[13];
h q[13];
h q[14];
rz(0.6843704085094304) q[14];
h q[14];
h q[15];
rz(0.2679799183618833) q[15];
h q[15];
h q[16];
rz(1.415458333005339) q[16];
h q[16];
h q[17];
rz(-1.0714666334868377) q[17];
h q[17];
h q[18];
rz(-0.14350155343754734) q[18];
h q[18];
h q[19];
rz(-1.0766325701343933) q[19];
h q[19];
rz(0.49870654187735314) q[0];
rz(-0.12254535325215031) q[1];
rz(0.8466212651703093) q[2];
rz(0.10640010825313972) q[3];
rz(0.08198832922029745) q[4];
rz(-0.16823968179826057) q[5];
rz(0.5298556720331722) q[6];
rz(0.27964936879858837) q[7];
rz(0.04963661934733001) q[8];
rz(-0.07841411043708824) q[9];
rz(-0.053863175012195376) q[10];
rz(-0.045839425733593855) q[11];
rz(0.04197624776647965) q[12];
rz(0.06269937776780651) q[13];
rz(0.04528007041472294) q[14];
rz(-0.011804936066036097) q[15];
rz(0.06659795973762819) q[16];
rz(-0.015915787177254524) q[17];
rz(0.01738392296196593) q[18];
rz(-0.06632934293312064) q[19];