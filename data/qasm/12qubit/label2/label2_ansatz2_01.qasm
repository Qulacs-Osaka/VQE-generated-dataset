OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.1170142050615164) q[0];
rz(0.6435920076116234) q[0];
ry(-0.07371163591465892) q[1];
rz(0.17042505755632573) q[1];
ry(-1.4512313162978805) q[2];
rz(-2.26262380108235) q[2];
ry(1.5707935932467247) q[3];
rz(3.1579273938540644e-05) q[3];
ry(7.555953884086362e-05) q[4];
rz(1.4623218335896395) q[4];
ry(-3.0922264159109485) q[5];
rz(-1.0466306967629624) q[5];
ry(0.0043187090014003005) q[6];
rz(-2.3896129991107133) q[6];
ry(0.0012145716047649465) q[7];
rz(-0.6912976743525414) q[7];
ry(-3.1415693046737627) q[8];
rz(-0.7197011880152449) q[8];
ry(5.277306591722208e-09) q[9];
rz(-0.8144707556173899) q[9];
ry(-3.7115395201681167e-09) q[10];
rz(-0.2501329983329601) q[10];
ry(-4.2492367378142406e-07) q[11];
rz(-0.10771778997049264) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.14159255840692) q[0];
rz(2.172704102002808) q[0];
ry(3.141592465899812) q[1];
rz(-1.401364318950015) q[1];
ry(3.95725692917966e-05) q[2];
rz(-2.4496860092085795) q[2];
ry(1.5707695598173776) q[3];
rz(1.5707161163373606) q[3];
ry(3.141592606631482) q[4];
rz(-1.113303827979047) q[4];
ry(-3.7518385083146206e-05) q[5];
rz(2.6173900572514) q[5];
ry(3.140389377819008) q[6];
rz(2.320534280624305) q[6];
ry(-0.003984368124122284) q[7];
rz(2.2422986168868464) q[7];
ry(3.0921754158197445) q[8];
rz(-2.891190544468875) q[8];
ry(3.141592648527788) q[9];
rz(-0.647834960885346) q[9];
ry(1.5708570552933767) q[10];
rz(-3.14117329961271) q[10];
ry(-1.6215608260385905) q[11];
rz(-2.8544446022183405) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.140368535183898) q[0];
rz(2.3097109169504013) q[0];
ry(-0.004328600195234066) q[1];
rz(-2.2956347517771523) q[1];
ry(-2.7151437327062644) q[2];
rz(2.4550496692927477) q[2];
ry(0.42404643187041824) q[3];
rz(-0.8763345838570958) q[3];
ry(-1.571409509989306) q[4];
rz(-0.011918779513580448) q[4];
ry(1.461855913596378) q[5];
rz(1.7931767771254565) q[5];
ry(0.07369654658463068) q[6];
rz(2.468701439273886) q[6];
ry(-3.117210309301418) q[7];
rz(-2.4708238092769785) q[7];
ry(3.141452501688623) q[8];
rz(-1.3197788742698018) q[8];
ry(1.5077725420074025e-09) q[9];
rz(-2.85629301250332) q[9];
ry(1.5722093458382984) q[10];
rz(1.5714006732247947) q[10];
ry(0.0014708049801453798) q[11];
rz(-1.8573406586750771) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.141591711859288) q[0];
rz(-2.545847944328553) q[0];
ry(-1.8176041946847422e-06) q[1];
rz(2.9545549143776646) q[1];
ry(-0.0007243693354151512) q[2];
rz(-0.33345013393952627) q[2];
ry(0.000746858341440948) q[3];
rz(-1.076079251794738) q[3];
ry(1.5734249602958892) q[4];
rz(-1.8002109944012707) q[4];
ry(3.129398888692931) q[5];
rz(-2.725065760501819) q[5];
ry(3.116978528082345) q[6];
rz(2.068659866298423) q[6];
ry(-3.067704369161631) q[7];
rz(-0.6323094699669971) q[7];
ry(1.4708823056910199) q[8];
rz(2.514309941716757) q[8];
ry(1.5707963143173804) q[9];
rz(-3.141592571207484) q[9];
ry(1.7586112414430615) q[10];
rz(-1.4636354428598182) q[10];
ry(-1.7528972258888098) q[11];
rz(-1.4592313527750136) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(4.961883970366898e-08) q[0];
rz(2.5330551356797235) q[0];
ry(1.2790503767234895e-07) q[1];
rz(-3.0220839149814767) q[1];
ry(-2.588194539674423e-08) q[2];
rz(-1.3440864185854982) q[2];
ry(2.2085026252227635e-08) q[3];
rz(2.72992793525059) q[3];
ry(-6.407063262159339e-07) q[4];
rz(2.577671005853701) q[4];
ry(-3.1415920232217553) q[5];
rz(-0.5991740861674373) q[5];
ry(3.1415926332423356) q[6];
rz(-2.764125832748492) q[6];
ry(2.1660534699208256e-08) q[7];
rz(2.0810202812413223) q[7];
ry(3.141592543996962) q[8];
rz(0.8214335826658488) q[8];
ry(1.5707964272881436) q[9];
rz(-1.6928766232914385) q[9];
ry(-6.395003524062304e-06) q[10];
rz(-0.22918945567810312) q[10];
ry(6.385000826725218e-06) q[11];
rz(2.9079989865171587) q[11];