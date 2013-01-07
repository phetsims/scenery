
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
    // constants
    var boxSizeRatio = 0.75;
    var boxTotalSize = 200;
    
    var Matrix3 = phet.math.Matrix3;
    
    var svgNS = 'http://www.w3.org/2000/svg';
    
    function buildEaselStage( main ) {
        var canvas = document.createElement( 'canvas' );
        canvas.id = 'easel-canvas';
        canvas.width = main.width();
        canvas.height = main.height();
        main.append( canvas );

        return new createjs.Stage( canvas );
    }
    
    function buildBaseContext( main ) {
        var baseCanvas = document.createElement( 'canvas' );
        baseCanvas.id = 'base-canvas';
        baseCanvas.width = main.width();
        baseCanvas.height = main.height();
        main.append( baseCanvas );
        
        return phet.canvas.initCanvas( baseCanvas );
    }
    
    function buildSVG( main ) {
        var svg = document.createElementNS( svgNS, 'svg' );
        svg.width = main.width();
        svg.height = main.height();
        main.append( svg );
        return svg;
    }
    
    phet.tests.easelVariableBox = function( main, resolution ) {
        var stage = buildEaselStage( main );
        var grid = new createjs.Container();
        
        var size = boxTotalSize;
        
        var boxRadius = 0.5 * boxSizeRatio * size / resolution;
        
        for( var row = 0; row < resolution; row++ ) {
            for( var col = 0; col < resolution; col++ ) {
                var shape = new createjs.Shape();
                shape.graphics.beginFill('rgba(255,0,0,1)').drawRect( -boxRadius, -boxRadius, boxRadius * 2, boxRadius * 2 );
                
                shape.x = ( col - ( resolution - 1 ) / 2 ) * size / resolution;
                shape.y = ( row - ( resolution - 1 ) / 2 ) * size / resolution;
                
                grid.addChild( shape );
            }
        }

        // center the grid        
        grid.x = main.width() / 2;
        grid.y = main.height() / 2;
        stage.addChild( grid );
        
        // return step function
        return function( timeElapsed ) {
            grid.rotation += timeElapsed * 180 / Math.PI;
            stage.update();
        }
    };

    phet.tests.customVariableBox = function( main, resolution ) {
        var baseContext = buildBaseContext( main );
        
        var xCenter = main.width() / 2;
        var yCenter = main.height() / 2;
        var rotation = 0;
        baseContext.fillStyle = 'rgba(255,0,0,1)';
        
        var size = boxTotalSize;
        
        var boxRadius = 0.5 * boxSizeRatio / resolution;
        var spotWidth = 1 / resolution;
        
        // TODO: optimize this
        var halfIshResolution = ( resolution - 1 ) / 2;
        
        // return step function
        return function( timeElapsed ) {
            // clear old location
            baseContext.clearRect( xCenter - 150, yCenter - 150, 300, 300 );
            // TODO: consider whether clearRect is faster if it's not under a transform!
            
            baseContext.save();
            rotation += timeElapsed;
            baseContext.translate( xCenter, yCenter );
            baseContext.rotate( rotation );
            baseContext.scale( size, size );
            
            baseContext.beginPath();
            
            for( var row = 0; row < resolution; row++ ) {
                var baseY = ( row - halfIshResolution ) / resolution;
                for( var col = 0; col < resolution; col++ ) {
                    var baseX = ( col - halfIshResolution ) / resolution;
                    baseContext.fillRect( baseX - boxRadius, baseY - boxRadius, boxRadius * 2, boxRadius * 2 );
                    // baseContext.moveTo( baseX - boxRadius, baseY - boxRadius );
                    // baseContext.lineTo( baseX + boxRadius, baseY - boxRadius );
                    // baseContext.lineTo( baseX + boxRadius, baseY + boxRadius );
                    // baseContext.lineTo( baseX - boxRadius, baseY + boxRadius );
                    // baseContext.lineTo( baseX - boxRadius, baseY - boxRadius );
                }
            }
            // baseContext.fill();
            
            baseContext.restore();
        }
    };
    
    phet.tests.sceneVariableBox = function( main, resolution ) {
        var size = boxTotalSize;
        var boxRadius = 0.5 * boxSizeRatio * size / resolution;
        
        var grid = new phet.scene.Node();
        
        grid.layerType = phet.scene.layers.CanvasLayer;
        
        for( var row = 0; row < resolution; row++ ) {
            for( var col = 0; col < resolution; col++ ) {
                grid.addChild( new phet.scene.nodes.Rectangle({
                    x: ( col - ( resolution - 1 ) / 2 ) * size / resolution - boxRadius,
                    y: ( row - ( resolution - 1 ) / 2 ) * size / resolution - boxRadius,
                    width: boxRadius * 2,
                    height: boxRadius * 2,
                    fill: 'rgba(255,0,0,1)'
                }) );
            }
        }
        
        // center the grid
        grid.translate( main.width() / 2, main.height() / 2 );
        
        // generate the layers
        grid.rebuildLayers( main );

        // return step function
        return function( timeElapsed ) {
            var bounds = grid.getBounds();
            // clear around another pixel or so, for antialiasing!
            grid._layerBeforeRender.context.clearRect( bounds.x() - 1, bounds.y() - 1, bounds.width() + 2, bounds.height() + 2 );
            
            grid.rotate( timeElapsed );
            
            // bounds validation here only necessary so we can show the black background behind it before rendering
            grid.validateBounds();
            
            // show a background behind the boxes that highlights the bounds
            bounds = grid.getBounds();
            var context = grid._layerBeforeRender.context;
            var tmpStyle = context.fillStyle;
            context.fillStyle = '#000';
            context.fillRect( bounds.x(), bounds.y(), bounds.width(), bounds.height() );
            context.fillStyle = tmpStyle;
            
            // TODO: dead region handling (this is a hack)
            // grid._layerBeforeRender.context.clearRect( main.width() / 2 - 150, main.height() / 2 - 150, 300, 300 );
            // baseContext.clearRect( 0, 0, main.width(), main.height() );
            grid.renderFull();
        }
    };
    
    phet.tests.svgVariableBox = function( main, resolution ) {
        var size = boxTotalSize;
        var boxRadius = 0.5 * boxSizeRatio * size / resolution;
        
        var svg = $( buildSVG( main ) );
        var group = $( document.createElementNS( svgNS, 'g' ) );
        group.attr( 'id', 'main-group' );
        group.attr( 'transform', 'translate(0,0)' );
        svg.append( group );
        
        for( var row = 0; row < resolution; row++ ) {
            for( var col = 0; col < resolution; col++ ) {
                var rect = $( document.createElementNS( svgNS, 'rect' ) );
                rect.attr( 'x', ( col - ( resolution - 1 ) / 2 ) * size / resolution - boxRadius );
                rect.attr( 'y', ( row - ( resolution - 1 ) / 2 ) * size / resolution - boxRadius );
                rect.attr( 'height', boxRadius * 2 );
                rect.attr( 'width', boxRadius * 2 );
                rect.css( 'fill', 'rgba(255,0,0,1)' );
                group.append( rect );
            }
        }
        
        var matrix = Matrix3.translation( main.width() / 2, main.height() / 2 );
        
        group[0].transform.baseVal.getItem( 0 ).setMatrix( matrix.toSVGMatrix() );
        
        // return step function
        return function( timeElapsed ) {
            matrix = matrix.timesMatrix( Matrix3.rotation2( timeElapsed ) );
            group[0].transform.baseVal.getItem( 0 ).setMatrix( matrix.toSVGMatrix() );
        }
    };
    
})();
