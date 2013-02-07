
(function(){
  
  marks.currentMarks = [
    {
      name: 'testA',
      count: 20,
      before: function() {
        this.width = 1024;
        this.height = 768;
        
        var main = $( '#main' );
        main.width( this.width );
        main.height( this.height );
        var scene = new phet.scene.Scene( main );
        for( var i = 0; i < 4000; i++ ) {
          scene.root.addChild( new phet.scene.Node( {
            shape: phet.scene.Shape.rectangle( ( Math.PI * i ) % this.width, ( 27 * i ) % this.height, 20, 20 ),
            fill: 'rgba(255,0,0,0.3)',
            stroke: '#000000'
          } ) );
        }
        
        this.main = main;
        this.scene = scene;
      },
      step: function() {
        this.scene.root.rotateAround( new phet.math.Vector2( this.width / 2, this.height / 2 ), 0.01 );
        this.scene.updateScene();
      },
      after: function() {
        this.main.empty();
      }
    },
    {
      name: 'testB',
      count: 20,
      before: function() {
        this.width = 1024;
        this.height = 768;
        
        var main = $( '#main' );
        main.width( this.width );
        main.height( this.height );
        var scene = new phet.scene.Scene( main );
        for( var i = 0; i < 4000; i++ ) {
          scene.root.addChild( new phet.scene.Node( {
            shape: phet.scene.Shape.rectangle( ( Math.PI * i ) % this.width, ( 27 * i ) % this.height, 20, 20 ),
            fill: 'rgba(0,0,255,0.3)',
            stroke: '#000000'
          } ) );
        }
        
        this.main = main;
        this.scene = scene;
      },
      step: function() {
        this.scene.root.rotateAround( new phet.math.Vector2( this.width / 2, this.height / 2 ), 0.01 );
        this.scene.updateScene();
      },
      after: function() {
        this.main.empty();
      }
    },
    {
      name: 'Empty Loop'
    }
  ];
  
})();
