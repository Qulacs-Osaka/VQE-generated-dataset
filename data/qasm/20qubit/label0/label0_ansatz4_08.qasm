OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(3.0908847633961853) q[0];
rz(-2.3522424198927436) q[0];
ry(-1.4765389076703193) q[1];
rz(-1.7617192004734212) q[1];
ry(0.0006474283951856492) q[2];
rz(1.5282064585307182) q[2];
ry(-0.011207910218779203) q[3];
rz(-2.9731628512374986) q[3];
ry(-0.2816748385143817) q[4];
rz(-1.7255436773422876) q[4];
ry(-1.308057859539302) q[5];
rz(3.044863334660481) q[5];
ry(-0.07318966984033014) q[6];
rz(1.7985021847002673) q[6];
ry(-1.084788792222514) q[7];
rz(-1.553226304226591) q[7];
ry(-3.1241636514492153) q[8];
rz(2.6845123213270488) q[8];
ry(0.0024009611239841173) q[9];
rz(-1.269206481778948) q[9];
ry(0.0028768483325603005) q[10];
rz(0.5704718228322961) q[10];
ry(0.00246639987034758) q[11];
rz(2.4097129470639644) q[11];
ry(-0.9282062609386363) q[12];
rz(-2.485894218183438) q[12];
ry(-1.4744849649470615) q[13];
rz(-3.055609096808327) q[13];
ry(0.01689609825701308) q[14];
rz(1.1933288355605522) q[14];
ry(-0.08042806604260855) q[15];
rz(-2.3193063480700293) q[15];
ry(0.0009611304007695055) q[16];
rz(-2.7179735748999443) q[16];
ry(-3.093942766894481) q[17];
rz(-1.61163218415492) q[17];
ry(0.054546161338709154) q[18];
rz(-1.7622147195641966) q[18];
ry(1.611694495717286) q[19];
rz(1.6195152383607057) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.9385009495937684) q[0];
rz(-0.25939846180329157) q[0];
ry(0.5658269018488864) q[1];
rz(-2.1000325817778744) q[1];
ry(-3.140661812583202) q[2];
rz(2.7711809479743232) q[2];
ry(0.0017489064082644985) q[3];
rz(-1.1056312082367599) q[3];
ry(-3.133066184954376) q[4];
rz(0.5713965204910302) q[4];
ry(2.0554762247174763) q[5];
rz(1.6603060853668266) q[5];
ry(3.1105582229580833) q[6];
rz(2.0129692726315453) q[6];
ry(1.1831021931178682) q[7];
rz(0.09176114508649547) q[7];
ry(1.6196359538185232) q[8];
rz(-2.563697739254146) q[8];
ry(-1.5415021210148292) q[9];
rz(0.5674075184855792) q[9];
ry(1.5620247843900854) q[10];
rz(1.3262604654570218) q[10];
ry(1.5545941842444222) q[11];
rz(-0.03831268617836008) q[11];
ry(-0.12832652753999818) q[12];
rz(2.5971717621354515) q[12];
ry(-1.4338835589361076) q[13];
rz(-0.04891712712800701) q[13];
ry(-0.43336855639098815) q[14];
rz(-2.30180008836074) q[14];
ry(-3.040426489546996) q[15];
rz(1.2925199735020054) q[15];
ry(-3.1347761909374237) q[16];
rz(-2.241914648812224) q[16];
ry(3.1372384947536895) q[17];
rz(-0.41855478794910805) q[17];
ry(-2.828481557859446) q[18];
rz(2.6393723452366413) q[18];
ry(1.4254055664534782) q[19];
rz(-1.1573307764041996) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.1048455817177643) q[0];
rz(1.0241446297173489) q[0];
ry(-1.9698725197120528) q[1];
rz(-0.5185667917958803) q[1];
ry(-0.015129401814923317) q[2];
rz(0.46255850838519835) q[2];
ry(3.137957678492498) q[3];
rz(-0.13037671839083168) q[3];
ry(0.04440157042378117) q[4];
rz(1.4149064684386345) q[4];
ry(2.0552698214343685) q[5];
rz(2.764227783585133) q[5];
ry(-0.0004159521428972468) q[6];
rz(0.2592454149913444) q[6];
ry(-0.0013152427204286) q[7];
rz(0.7358440312965346) q[7];
ry(-0.11294967196412774) q[8];
rz(-0.8284739740217597) q[8];
ry(-0.09275891517870072) q[9];
rz(2.581416027153221) q[9];
ry(1.6707264240890662) q[10];
rz(1.9464139358306987) q[10];
ry(1.1062008842410265) q[11];
rz(-0.9931885440504609) q[11];
ry(0.9398933955913638) q[12];
rz(1.4845925444101473) q[12];
ry(0.006962756029028307) q[13];
rz(-0.028228872471978257) q[13];
ry(1.3443137684786697) q[14];
rz(-0.7306552177402414) q[14];
ry(-3.072589651146688) q[15];
rz(0.2618181853027953) q[15];
ry(3.1401384224372606) q[16];
rz(0.94937598062016) q[16];
ry(-3.0914627664741787) q[17];
rz(-0.9556457069484208) q[17];
ry(-1.4448810320791328) q[18];
rz(1.3660210662151657) q[18];
ry(1.9229577372586588) q[19];
rz(-2.2237405535974757) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.0535744939472114) q[0];
rz(-1.455510136811226) q[0];
ry(3.0181621655996183) q[1];
rz(0.49329088521141706) q[1];
ry(-0.372688146091658) q[2];
rz(3.0118453911080043) q[2];
ry(-3.1053248988627233) q[3];
rz(2.737797929065114) q[3];
ry(-0.07371680632582844) q[4];
rz(-0.6940512541759768) q[4];
ry(-0.1254458329022041) q[5];
rz(0.34619954819754994) q[5];
ry(3.1071675758140045) q[6];
rz(-2.6232225160120644) q[6];
ry(2.682235527432199) q[7];
rz(0.3980498656480075) q[7];
ry(-1.5227362698941531) q[8];
rz(-2.5189751914264162) q[8];
ry(1.5664195199100701) q[9];
rz(1.3779097717599371) q[9];
ry(3.113171826848415) q[10];
rz(-0.197284607848216) q[10];
ry(-0.011803787593444668) q[11];
rz(-2.394691181104895) q[11];
ry(1.5518810942906291) q[12];
rz(1.572838720096276) q[12];
ry(-1.572597597486924) q[13];
rz(-1.5452711604892457) q[13];
ry(0.04473470347029718) q[14];
rz(-2.278114551169752) q[14];
ry(-0.00043906051579956085) q[15];
rz(-2.958844812042805) q[15];
ry(-0.002018152249140037) q[16];
rz(2.2144038526842573) q[16];
ry(3.130082397359693) q[17];
rz(-2.1475857846658095) q[17];
ry(-0.05202908809473694) q[18];
rz(1.4306149402707948) q[18];
ry(2.2823322182081447) q[19];
rz(-1.2506428403589434) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.11436903605325795) q[0];
rz(-0.25401559127116613) q[0];
ry(0.6933472326885262) q[1];
rz(0.259926397809461) q[1];
ry(0.030013999987945662) q[2];
rz(-1.8596847068490758) q[2];
ry(3.064813338492622) q[3];
rz(0.332707293766771) q[3];
ry(-0.00042218457502688725) q[4];
rz(-0.15722211320179968) q[4];
ry(-3.1383562264543725) q[5];
rz(1.4678487925000554) q[5];
ry(-1.244414379589161) q[6];
rz(1.9580995377062123) q[6];
ry(-1.4291919690073636) q[7];
rz(1.3381404188823607) q[7];
ry(0.014556358857721996) q[8];
rz(-0.020855893285484765) q[8];
ry(-0.06966816613137411) q[9];
rz(-1.0752794090658724) q[9];
ry(-3.1401786136092538) q[10];
rz(-0.17948813216128953) q[10];
ry(0.0021608983159397965) q[11];
rz(-1.2957075969454124) q[11];
ry(-1.6186560490844677) q[12];
rz(-0.6942541841073817) q[12];
ry(2.506268433417329) q[13];
rz(-2.21412785382537) q[13];
ry(-3.1014782184588) q[14];
rz(-2.6598222120203014) q[14];
ry(2.9141660460488152) q[15];
rz(0.15996272037872086) q[15];
ry(3.140667598390284) q[16];
rz(-2.1837607121950104) q[16];
ry(-3.123403970286962) q[17];
rz(2.0936694575577954) q[17];
ry(1.690284549391626) q[18];
rz(1.5676024276225675) q[18];
ry(-1.8531287178298845) q[19];
rz(-1.8816389670164906) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.12515434185505825) q[0];
rz(-0.32985917704456885) q[0];
ry(3.025432669537936) q[1];
rz(-1.214427187215723) q[1];
ry(-0.06297992596395863) q[2];
rz(-1.702301443160056) q[2];
ry(-1.6476751703066863) q[3];
rz(-3.119083717606388) q[3];
ry(-3.117346858675709) q[4];
rz(2.834564395520638) q[4];
ry(-2.987568221347801) q[5];
rz(-0.04193439284129674) q[5];
ry(1.4325824271951833) q[6];
rz(-1.5881706348598317) q[6];
ry(-0.308126441790642) q[7];
rz(0.24780930741729162) q[7];
ry(0.0063881979123463495) q[8];
rz(1.6598609548322423) q[8];
ry(-0.0007970812671924676) q[9];
rz(-1.8299252167709446) q[9];
ry(2.119540572605306) q[10];
rz(-0.022794996147006797) q[10];
ry(1.563514163834095) q[11];
rz(0.1844694886737095) q[11];
ry(-3.131204198611746) q[12];
rz(0.5930998385740978) q[12];
ry(-0.01737249906210536) q[13];
rz(1.264814810852477) q[13];
ry(-1.5241462355328448) q[14];
rz(-2.814900282117584) q[14];
ry(-0.009586758394979356) q[15];
rz(-1.7848488649349852) q[15];
ry(1.5092189934190214) q[16];
rz(-0.21295147008618615) q[16];
ry(0.13030678651853791) q[17];
rz(-1.2176217660850985) q[17];
ry(-2.154077283889827) q[18];
rz(0.28923185130514595) q[18];
ry(-3.033039144177854) q[19];
rz(1.7703014637236014) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.0425040439350375) q[0];
rz(1.466256844956007) q[0];
ry(-3.0933004359080565) q[1];
rz(2.336906074576454) q[1];
ry(-0.0009759223266314621) q[2];
rz(2.844559028625454) q[2];
ry(-0.05814118986476302) q[3];
rz(1.8775794481363628) q[3];
ry(-0.051489886246177896) q[4];
rz(-3.094334691460434) q[4];
ry(-3.1015873500934186) q[5];
rz(-2.9010662208161455) q[5];
ry(-1.6923885320191603) q[6];
rz(1.5633413534381406) q[6];
ry(-0.302383005716087) q[7];
rz(-2.9262776296898165) q[7];
ry(-3.1357200471302393) q[8];
rz(-0.6923389131334633) q[8];
ry(-3.141278003093588) q[9];
rz(-1.357969835922618) q[9];
ry(0.23803211332295415) q[10];
rz(0.040417113347376805) q[10];
ry(3.1397640329621535) q[11];
rz(0.40317334903974145) q[11];
ry(0.2559942621773681) q[12];
rz(0.799272991575104) q[12];
ry(-3.1386350161695087) q[13];
rz(2.976284302626287) q[13];
ry(0.0021044853338860747) q[14];
rz(-1.4911290359950966) q[14];
ry(2.886207034920785) q[15];
rz(2.178371372795567) q[15];
ry(6.421971632786239e-05) q[16];
rz(-1.242721068215305) q[16];
ry(-0.0002323770876245444) q[17];
rz(-1.45335031717513) q[17];
ry(-1.5177115934118255) q[18];
rz(1.6143335601481876) q[18];
ry(1.5716944440858) q[19];
rz(-0.9771242081753575) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.044298333597383166) q[0];
rz(1.6470710536679667) q[0];
ry(0.044015236192248786) q[1];
rz(-2.2307883413225724) q[1];
ry(-0.004729823541031131) q[2];
rz(0.8960489010113601) q[2];
ry(-0.0062275512997027604) q[3];
rz(1.1870526836917135) q[3];
ry(-1.239457714343929) q[4];
rz(1.5445553893670478) q[4];
ry(-0.09102526130695118) q[5];
rz(-1.826095025637853) q[5];
ry(-1.868721660206381) q[6];
rz(0.7298969844465699) q[6];
ry(0.08811700926785981) q[7];
rz(2.261753973258319) q[7];
ry(1.5727600922553586) q[8];
rz(-0.877044081661661) q[8];
ry(-1.331141780479097) q[9];
rz(1.264298542572378) q[9];
ry(2.574446266751568) q[10];
rz(-3.1301238710271257) q[10];
ry(3.131203837433067) q[11];
rz(0.22184187841249048) q[11];
ry(-3.13963973214924) q[12];
rz(2.926024995510323) q[12];
ry(-3.1274355315820697) q[13];
rz(2.4181971357546974) q[13];
ry(-1.5385290329216836) q[14];
rz(-0.5057129952109704) q[14];
ry(3.137428308717335) q[15];
rz(-0.2005045434970004) q[15];
ry(0.004207416050739087) q[16];
rz(1.1458789578348343) q[16];
ry(2.8801787686094493) q[17];
rz(1.8921506648379447) q[17];
ry(-2.7885481982158233) q[18];
rz(-1.5129966060193856) q[18];
ry(-0.013679180389433832) q[19];
rz(2.4055782560281216) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.10801162485872061) q[0];
rz(1.980185909512163) q[0];
ry(3.079838753261919) q[1];
rz(-1.6544863990477285) q[1];
ry(0.013653463631396967) q[2];
rz(-2.9414842553663325) q[2];
ry(-0.23720765767067725) q[3];
rz(0.18213744605397153) q[3];
ry(-0.05368642106491021) q[4];
rz(0.8558805691884784) q[4];
ry(0.06099539693799238) q[5];
rz(-2.6807594689631338) q[5];
ry(0.0009282451751776049) q[6];
rz(2.08468215624791) q[6];
ry(3.141583333001502) q[7];
rz(2.1044162030830975) q[7];
ry(3.139540121861978) q[8];
rz(1.8773067300146327) q[8];
ry(-3.1367760932924402) q[9];
rz(-0.34637533072626514) q[9];
ry(1.5047002232775777) q[10];
rz(-0.000682700237677874) q[10];
ry(2.2739122792731012) q[11];
rz(0.21747533713427106) q[11];
ry(-0.007845531526272431) q[12];
rz(-2.1247833481178464) q[12];
ry(1.631297567062792) q[13];
rz(-3.0953154062543304) q[13];
ry(3.140599398105573) q[14];
rz(-2.3988802642624965) q[14];
ry(-3.1294141563733784) q[15];
rz(2.116645854352771) q[15];
ry(3.141407303596401) q[16];
rz(-0.42137317113911593) q[16];
ry(-3.1383750654786855) q[17];
rz(1.8812427040303419) q[17];
ry(-1.6038386400698925) q[18];
rz(-1.5887686680722013) q[18];
ry(1.5860136930894073) q[19];
rz(1.5421914936094994) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1216600127333396) q[0];
rz(-3.0820619226218007) q[0];
ry(1.6169409486018358) q[1];
rz(-1.5882390883020419) q[1];
ry(-3.0689670081558362) q[2];
rz(2.754193214953332) q[2];
ry(-3.1297288667485126) q[3];
rz(-2.635003556760086) q[3];
ry(-1.3314302347866738) q[4];
rz(-2.51448538140964) q[4];
ry(-1.8959836607853449) q[5];
rz(2.454171097676961) q[5];
ry(-0.018965522576630405) q[6];
rz(0.2329807461666166) q[6];
ry(2.963382447593782) q[7];
rz(1.2107353257604492) q[7];
ry(-0.10621198988683593) q[8];
rz(-0.20539415222369908) q[8];
ry(2.75077862571432) q[9];
rz(2.8570091780308884) q[9];
ry(-2.878483351052609) q[10];
rz(-1.572155515526922) q[10];
ry(-0.005541651535751335) q[11];
rz(1.3553380546954257) q[11];
ry(0.02188621768968879) q[12];
rz(-1.7541042533460482) q[12];
ry(0.006307223192161141) q[13];
rz(1.5266975223305717) q[13];
ry(0.012410528798517896) q[14];
rz(0.540104077161134) q[14];
ry(-3.113737550738606) q[15];
rz(3.1238948849439008) q[15];
ry(-2.9777231629658685) q[16];
rz(-2.7134869703390496) q[16];
ry(-2.005725624310614) q[17];
rz(-0.18500353531815628) q[17];
ry(0.005982043149848515) q[18];
rz(0.769886236074738) q[18];
ry(1.2427239979312246) q[19];
rz(-3.1367791540851853) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1391675679600435) q[0];
rz(0.17693129515966036) q[0];
ry(-1.5550710773394045) q[1];
rz(-0.013271380861205275) q[1];
ry(0.0027220736408253288) q[2];
rz(-2.7009983388403764) q[2];
ry(0.0006967467632438117) q[3];
rz(2.7930146730098686) q[3];
ry(-0.043415399666137056) q[4];
rz(-1.9604558932624103) q[4];
ry(3.0743083459472373) q[5];
rz(1.5438932355848545) q[5];
ry(-3.1415641176046845) q[6];
rz(1.3695714289961618) q[6];
ry(3.1415151931561667) q[7];
rz(2.9708251869549205) q[7];
ry(-3.140736147848079) q[8];
rz(0.9463160831305972) q[8];
ry(-3.1374234785484187) q[9];
rz(-1.6091525804630955) q[9];
ry(1.5669370916840364) q[10];
rz(1.5923292186262268) q[10];
ry(1.5678349447699849) q[11];
rz(-0.7624118010128746) q[11];
ry(1.5616834076128927) q[12];
rz(1.4911974937272596) q[12];
ry(-1.5688379527567027) q[13];
rz(-1.5896370201651449) q[13];
ry(0.0009089691899821703) q[14];
rz(-1.4116921671076836) q[14];
ry(-3.141313379040267) q[15];
rz(1.7446203739968442) q[15];
ry(0.001076374702690508) q[16];
rz(-0.6690945945191754) q[16];
ry(3.138353115318467) q[17];
rz(-1.7584671773926894) q[17];
ry(0.003183223897029147) q[18];
rz(2.412564861274242) q[18];
ry(-0.4187472103571322) q[19];
rz(2.236351374112413) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.570157627635596) q[0];
rz(0.0026256447608359323) q[0];
ry(0.3114472193163822) q[1];
rz(1.5950499632909174) q[1];
ry(-1.5533536777388477) q[2];
rz(0.015803215056489905) q[2];
ry(1.5577192789167675) q[3];
rz(-1.4800370977283268) q[3];
ry(-0.8533532258534522) q[4];
rz(2.4637156576780717) q[4];
ry(0.49489476339295724) q[5];
rz(0.8135528089620877) q[5];
ry(0.019252765882227685) q[6];
rz(-1.7860852989068983) q[6];
ry(-1.6562307724017202) q[7];
rz(2.8894600277327904) q[7];
ry(-1.5670207438435093) q[8];
rz(-1.58185310706829) q[8];
ry(3.129617619959872) q[9];
rz(-2.938364477089299) q[9];
ry(1.6142777447880536) q[10];
rz(3.095503104274411) q[10];
ry(1.5809823660019717) q[11];
rz(1.5954714493650255) q[11];
ry(-0.1829830503402574) q[12];
rz(-3.051615461286075) q[12];
ry(1.6064037639864441) q[13];
rz(-1.5582867206685298) q[13];
ry(3.124991749792197) q[14];
rz(0.37623227181717395) q[14];
ry(1.581951805870487) q[15];
rz(-3.108833615601542) q[15];
ry(0.14776937528162737) q[16];
rz(-1.3991607776451678) q[16];
ry(1.5582487379429142) q[17];
rz(-1.1053939824264256) q[17];
ry(1.5739080576170834) q[18];
rz(-0.0035950029146149297) q[18];
ry(-3.1063105894843175) q[19];
rz(-2.480765282819131) q[19];