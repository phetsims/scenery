
(function(){
  
  marks.currentMarks = [
    {
      name: 'Rotated group of squares with hardcoded xy',
      count: 20,
      before: function() {
        this.width = 1024;
        this.height = 768;
        
        var main = $( '#main' );
        main.width( this.width );
        main.height( this.height );
        var scene = new scenery.Scene( main );
        for( var i = 0; i < 4000; i++ ) {
          scene.root.addChild( new scenery.Node( {
            shape: scenery.Shape.rectangle( ( Math.PI * i ) % this.width, ( 27 * i ) % this.height, 20, 20 ),
            fill: 'rgba(255,0,0,0.3)',
            stroke: '#000000'
          } ) );
        }
        
        this.main = main;
        this.scene = scene;
      },
      step: function() {
        this.scene.root.rotateAround( new phet.math.Vector2( this.width / 2, this.height / 2 ), 0.1 );
        this.scene.updateScene();
      },
      after: function() {
        this.main.empty();
      }
    },
    {
      name: 'Rotated group of squares with transform xy',
      count: 20,
      before: function() {
        this.width = 1024;
        this.height = 768;
        
        var main = $( '#main' );
        main.width( this.width );
        main.height( this.height );
        var scene = new scenery.Scene( main );
        for( var i = 0; i < 4000; i++ ) {
          scene.root.addChild( new scenery.Node( {
            shape: scenery.Shape.rectangle( 0, 0, 20, 20 ),
            fill: 'rgba(0,0,255,0.3)',
            stroke: '#000000',
            x: ( Math.PI * i ) % this.width,
            y: ( 27 * i ) % this.height
          } ) );
        }
        
        this.main = main;
        this.scene = scene;
      },
      step: function() {
        this.scene.root.rotateAround( new phet.math.Vector2( this.width / 2, this.height / 2 ), 0.1 );
        this.scene.updateScene();
      },
      after: function() {
        this.main.empty();
      }
    },
    {
      name: 'Individually Rotated Squares',
      count: 20,
      before: function() {
        this.width = 1024;
        this.height = 768;
        
        var main = $( '#main' );
        main.width( this.width );
        main.height( this.height );
        var scene = new scenery.Scene( main );
        for( var i = 0; i < 4000; i++ ) {
          scene.root.addChild( new scenery.Node( {
            shape: scenery.Shape.rectangle( 0, 0, 20, 20 ),
            fill: 'rgba(0,255,0,0.3)',
            stroke: '#000000',
            x: ( Math.PI * i ) % this.width,
            y: ( 27 * i ) % this.height
          } ) );
        }
        
        this.main = main;
        this.scene = scene;
      },
      step: function() {
        _.each( this.scene.root.getChildren(), function( child ) {
          child.rotate( 0.1 );
        } );
        this.scene.updateScene();
      },
      after: function() {
        this.main.empty();
      }
    },
    {
      name: 'Adding Hexagons',
      count: 20,
      before: function() {
        this.width = 1024;
        this.height = 768;
        
        this.x = 0;
        this.y = 0;
        
        var main = $( '#main' );
        main.width( this.width );
        main.height( this.height );
        var scene = new scenery.Scene( main );
        
        this.main = main;
        this.scene = scene;
      },
      step: function() {
        for ( var i = 0; i < 200; i++ ) {
          this.scene.root.addChild( new scenery.Node( {
            shape: scenery.Shape.regularPolygon( 6, 22 ),
            fill: 'rgba(255,0,255,0.3)',
            stroke: '#000000',
            x: this.x,
            y: this.y
          } ) );
          this.x = ( this.x + 213 ) % this.width;
          this.y = ( this.y + 377 ) % this.height;
        }
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
