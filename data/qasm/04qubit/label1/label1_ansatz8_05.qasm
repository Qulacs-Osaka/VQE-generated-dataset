OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.0355202164659114) q[0];
ry(-2.678294102758047) q[1];
cx q[0],q[1];
ry(-3.0354736426160116) q[0];
ry(2.22258587252654) q[1];
cx q[0],q[1];
ry(-1.1301948349226973) q[2];
ry(2.1901665650718702) q[3];
cx q[2],q[3];
ry(0.053446952769595275) q[2];
ry(-2.766853922878814) q[3];
cx q[2],q[3];
ry(0.37055125807842354) q[0];
ry(2.365129128551319) q[2];
cx q[0],q[2];
ry(-0.4841528206793875) q[0];
ry(-3.0338495592728463) q[2];
cx q[0],q[2];
ry(-0.10410338894420766) q[1];
ry(0.06369769057378608) q[3];
cx q[1],q[3];
ry(1.0448650815767326) q[1];
ry(2.094351317293224) q[3];
cx q[1],q[3];
ry(2.2312179141459536) q[0];
ry(0.17013680576270487) q[1];
cx q[0],q[1];
ry(2.845168099986069) q[0];
ry(-2.867250387419066) q[1];
cx q[0],q[1];
ry(2.264980573174787) q[2];
ry(-1.6558452452778827) q[3];
cx q[2],q[3];
ry(0.19891830183268944) q[2];
ry(-2.4358027103615107) q[3];
cx q[2],q[3];
ry(-1.0695582620493727) q[0];
ry(-0.5856901349152743) q[2];
cx q[0],q[2];
ry(-2.6575400646779572) q[0];
ry(-2.205747698611571) q[2];
cx q[0],q[2];
ry(-1.1777063102277987) q[1];
ry(-2.745873493332515) q[3];
cx q[1],q[3];
ry(1.668331607892159) q[1];
ry(0.590384371963733) q[3];
cx q[1],q[3];
ry(-1.985069816237572) q[0];
ry(-0.17224375638564815) q[1];
cx q[0],q[1];
ry(1.0927791461336145) q[0];
ry(1.5625045993047841) q[1];
cx q[0],q[1];
ry(0.07587114272503204) q[2];
ry(-2.0292916024494376) q[3];
cx q[2],q[3];
ry(-2.1420072123350957) q[2];
ry(-1.6243649287412492) q[3];
cx q[2],q[3];
ry(-1.0428954184787245) q[0];
ry(0.3114272856183509) q[2];
cx q[0],q[2];
ry(-1.303130319878803) q[0];
ry(-2.2252144347859195) q[2];
cx q[0],q[2];
ry(-0.7999849185617569) q[1];
ry(-1.4830848741535752) q[3];
cx q[1],q[3];
ry(2.5139861772806444) q[1];
ry(-2.1038408243785813) q[3];
cx q[1],q[3];
ry(2.4243074298801326) q[0];
ry(2.443378288518163) q[1];
cx q[0],q[1];
ry(-0.051972193112580285) q[0];
ry(0.5628195985363228) q[1];
cx q[0],q[1];
ry(-0.3491819568545642) q[2];
ry(-2.4077442059021714) q[3];
cx q[2],q[3];
ry(2.3990795994097955) q[2];
ry(-3.0065732006790915) q[3];
cx q[2],q[3];
ry(1.0334607034365568) q[0];
ry(-1.4756863809176846) q[2];
cx q[0],q[2];
ry(-0.34956812395158476) q[0];
ry(-0.9565403321856303) q[2];
cx q[0],q[2];
ry(1.8149169907186646) q[1];
ry(0.4120857425457132) q[3];
cx q[1],q[3];
ry(1.7206636893344236) q[1];
ry(0.8749694389745843) q[3];
cx q[1],q[3];
ry(-1.9855125496349881) q[0];
ry(-0.7095975573876343) q[1];
cx q[0],q[1];
ry(0.0740726526731521) q[0];
ry(-2.9987198732248244) q[1];
cx q[0],q[1];
ry(-2.568832817078294) q[2];
ry(-1.0955669250345235) q[3];
cx q[2],q[3];
ry(0.052677823348470376) q[2];
ry(-1.207942115817564) q[3];
cx q[2],q[3];
ry(1.1189665589493556) q[0];
ry(-2.9294418825307904) q[2];
cx q[0],q[2];
ry(-0.43071024891808457) q[0];
ry(-1.5902181245847578) q[2];
cx q[0],q[2];
ry(0.7187876901938538) q[1];
ry(-2.1278697443361607) q[3];
cx q[1],q[3];
ry(2.879561249159811) q[1];
ry(2.9034110462762324) q[3];
cx q[1],q[3];
ry(1.6831448928911523) q[0];
ry(-0.12014319316763) q[1];
cx q[0],q[1];
ry(-2.2230119647590074) q[0];
ry(-2.225305277061751) q[1];
cx q[0],q[1];
ry(0.06699526761388874) q[2];
ry(-0.9339951540442177) q[3];
cx q[2],q[3];
ry(-2.4836444205976558) q[2];
ry(1.313510108175344) q[3];
cx q[2],q[3];
ry(2.8177418856074796) q[0];
ry(2.8177391935107963) q[2];
cx q[0],q[2];
ry(-0.3845068920240863) q[0];
ry(-1.7418623215233415) q[2];
cx q[0],q[2];
ry(-0.36904346733738436) q[1];
ry(-2.111651572882598) q[3];
cx q[1],q[3];
ry(1.6284623833200518) q[1];
ry(0.017593212505223353) q[3];
cx q[1],q[3];
ry(1.7854757417082399) q[0];
ry(-2.0581594590067223) q[1];
cx q[0],q[1];
ry(1.746051009656259) q[0];
ry(-0.7211992144334944) q[1];
cx q[0],q[1];
ry(0.13003755235902087) q[2];
ry(1.2780612411801389) q[3];
cx q[2],q[3];
ry(2.279766401048197) q[2];
ry(1.4985399537346318) q[3];
cx q[2],q[3];
ry(0.30534858597095715) q[0];
ry(0.3359377606913885) q[2];
cx q[0],q[2];
ry(-3.01841956407261) q[0];
ry(0.40437041796657525) q[2];
cx q[0],q[2];
ry(1.860957757399838) q[1];
ry(0.31959552695994553) q[3];
cx q[1],q[3];
ry(-1.2833288082490286) q[1];
ry(-0.44981076386657204) q[3];
cx q[1],q[3];
ry(2.3161345853981303) q[0];
ry(-0.5203608660779486) q[1];
cx q[0],q[1];
ry(-1.1562676905590215) q[0];
ry(0.8600387545635533) q[1];
cx q[0],q[1];
ry(1.5030433080878947) q[2];
ry(-0.2520739112326817) q[3];
cx q[2],q[3];
ry(-2.5139367064387734) q[2];
ry(0.676165535482413) q[3];
cx q[2],q[3];
ry(-2.854685821231802) q[0];
ry(-1.0805270608086754) q[2];
cx q[0],q[2];
ry(1.7872544798977867) q[0];
ry(2.06759563574984) q[2];
cx q[0],q[2];
ry(2.122629607784437) q[1];
ry(2.202247571299658) q[3];
cx q[1],q[3];
ry(-0.39455611453225053) q[1];
ry(-0.6810879535203934) q[3];
cx q[1],q[3];
ry(-1.226734621757891) q[0];
ry(-2.530584364629429) q[1];
ry(2.002648248177538) q[2];
ry(1.088732417074425) q[3];