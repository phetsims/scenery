// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    
    // TODO: remove flag and tests after we're done
    var debugChromeBoundsScanning = false;
    
    var Matrix3 = phet.math.Matrix3;
    var Transform3 = phet.math.Transform3;
    var Bounds2 = phet.math.Bounds2;
    var Vector2 = phet.math.Vector2;
    
    function p( x, y ) {
        return new Vector2( x, y );
    }
    
    /*---------------------------------------------------------------------------*
    * Slow-but-accurate text bounds
    *----------------------------------------------------------------------------*/        
    
    // drawingStyles should include font, textAlign, textBaseline, direction
    // textAlign = 'left', textBaseline = 'alphabetic' and direction = 'ltr' are recommended
	phet.scene.canvasTextBoundsAccurate = function( text, fontDrawingStyles ) {
        return canvasAccurateBounds( function( context ) {
            context.font = fontDrawingStyles.font;
            context.textAlign = fontDrawingStyles.textAlign;
            context.textBaseline = fontDrawingStyles.textBaseline;
            context.direction = fontDrawingStyles.direction;
            context.fillText( text, 0, 0 );
        } );
    };
    
    // given a data snapshot and transform, calculate range on how large / small the bounds can be
    // very conservative, with an effective 1px extra range to allow for differences in anti-aliasing
    // for performance concerns, this does not support skews / rotations / anything but translation and scaling
    phet.scene.scanBounds = function( imageData, resolution, transform ) {
        
        // entry will be true if any pixel with the given x or y value is non-rgba(0,0,0,0)
        var dirtyX = _.map( _.range( resolution ), function() { return false; } );
        var dirtyY = _.map( _.range( resolution ), function() { return false; } );
        
        for( var x = 0; x < resolution; x++ ) {
            for( var y = 0; y < resolution; y++ ) {
                var offset = 4 * ( y * resolution + x );
                if( imageData.data[offset] != 0 || imageData.data[offset+1] != 0 || imageData.data[offset+2] != 0 || imageData.data[offset+3] != 0 ) {
                    dirtyX[x] = true;
                    dirtyY[y] = true;
                }
            }
        }
        
        var minX = _.indexOf( dirtyX, true );
        var maxX = _.lastIndexOf( dirtyX, true );
        var minY = _.indexOf( dirtyY, true );
        var maxY = _.lastIndexOf( dirtyY, true );
        
        // based on pixel boundaries. for minBounds, the inner edge of the dirty pixel. for maxBounds, the outer edge of the adjacent non-dirty pixel
        // results in a spread of 2 for the identity transform (or any translated form)
        var extraSpread = resolution / 16; // is Chrome antialiasing really like this? dear god... TODO!!!
        return {
            minBounds: new Bounds2(
                ( minX < 1 || minX >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( minX + 1 + extraSpread, 0 ) ).x,
                ( minY < 1 || minY >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( 0, minY + 1 + extraSpread ) ).y,
                ( maxX < 1 || maxX >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( maxX - extraSpread, 0 ) ).x,
                ( maxY < 1 || maxY >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( 0, maxY - extraSpread ) ).y
            ),
            maxBounds: new Bounds2(
                ( minX < 1 || minX >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( minX - 1 - extraSpread, 0 ) ).x,
                ( minY < 1 || minY >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( 0, minY - 1 - extraSpread ) ).y,
                ( maxX < 1 || maxX >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( maxX + 2 + extraSpread, 0 ) ).x,
                ( maxY < 1 || maxY >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( 0, maxY + 2 + extraSpread ) ).y
            )
        };
    };
    
    phet.scene.canvasAccurateBounds = function( renderToContext, options ) {
        // how close to the actual bounds do we need to be?
        var precision = ( options && options.precision ) ? options.precision : 0.001;
        
        // 512x512 default square resolution
        var resolution = ( options && options.resolution ) ? options.resolution : 128;
        
        // at 1/16x default, we want to be able to get the bounds accurately for something as large as 16x our initial resolution
        // divisible by 2 so hopefully we avoid more quirks from Canvas rendering engines
        var initialScale = ( options && options.initialScale ) ? options.initialScale : (1/16);
        
        var minBounds = Bounds2.NOTHING;
        var maxBounds = Bounds2.EVERYTHING;
        
        var canvas = document.createElement( 'canvas' );
        canvas.width = resolution;
        canvas.height = resolution;
        var context = phet.canvas.initCanvas( canvas );
        
        if( debugChromeBoundsScanning ) {
            $( window ).ready( function() {
                var header = document.createElement( 'h2' );
                $( header ).text( 'Bounds Scan' );
                $( '#display' ).append( header );
            } );
        }
        
        function scan( transform ) {
            // save/restore, in case the render tries to do any funny stuff like clipping, etc.
            context.save();
            transform.matrix.canvasSetTransform( context );
            renderToContext( context );
            context.restore();
            
            var data = context.getImageData( 0, 0, resolution, resolution );
            var minMaxBounds = phet.scene.scanBounds( data, resolution, transform );
            
            // TODO: remove after debug
            if( debugChromeBoundsScanning ) {
                function snapshotToCanvas( snapshot ) {
                    var canvas = document.createElement( 'canvas' );
                    canvas.width = resolution;
                    canvas.height = resolution;
                    var context = phet.canvas.initCanvas( canvas );
                    context.putImageData( snapshot, 0, 0 );
                    $( canvas ).css( 'border', '1px solid black' );
                    $( window ).ready( function() {
                        //$( '#display' ).append( $( document.createElement( 'div' ) ).text( 'Bounds: ' +  ) );
                        $( '#display' ).append( canvas );
                    } );
                }
                snapshotToCanvas( data );
            }
            
            context.clearRect( 0, 0, resolution, resolution );
            
            return minMaxBounds;
        }
        
        // attempts to map the bounds specified to the entire testing canvas (minus a fine border), so we can nail down the location quickly
        function idealTransform( bounds ) {
            // so that the bounds-edge doesn't land squarely on the boundary
            var borderSize = 2;
            
            var scaleX = ( resolution - borderSize * 2 ) / ( bounds.xMax - bounds.xMin );
            var scaleY = ( resolution - borderSize * 2 ) / ( bounds.yMax - bounds.yMin );
            var translationX = -scaleX * bounds.xMin + borderSize;
            var translationY = -scaleY * bounds.yMin + borderSize;
            
            return new Transform3( Matrix3.translation( translationX, translationY ).timesMatrix( Matrix3.scaling( scaleX, scaleY ) ) );
        }
        
        var initialTransform = new Transform3( );
        // make sure to initially center our object, so we don't miss the bounds
        initialTransform.append( Matrix3.translation( resolution / 2, resolution / 2 ) );
        initialTransform.append( Matrix3.scaling( initialScale ) );
        
        var coarseBounds = scan( initialTransform );
        
        minBounds = minBounds.union( coarseBounds.minBounds );
        maxBounds = maxBounds.intersection( coarseBounds.maxBounds );
        
        var tempMin, tempMax;
        
        // xMin
        tempMin = maxBounds.yMin;
        tempMax = maxBounds.yMax;
        while( isFinite( minBounds.xMin ) && isFinite( maxBounds.xMin ) && Math.abs( minBounds.xMin - maxBounds.xMin ) > precision ) {
            // use maximum bounds except for the x direction, so we don't miss things that we are looking for
            var refinedBounds = scan( idealTransform( new Bounds2( maxBounds.xMin, tempMin, minBounds.xMin, tempMax ) ) );
            
            if( minBounds.xMin <= refinedBounds.minBounds.xMin && maxBounds.xMin >= refinedBounds.maxBounds.xMin ) {
                // sanity check - break out of an infinite loop!
                console.log( 'warning, exiting infinite loop!' );
                console.log( 'transformed "min" xMin: ' + idealTransform( new Bounds2( maxBounds.xMin, maxBounds.yMin, minBounds.xMin, maxBounds.yMax ) ).transformPosition2( p( minBounds.xMin, 0 ) ) );
                console.log( 'transformed "max" xMin: ' + idealTransform( new Bounds2( maxBounds.xMin, maxBounds.yMin, minBounds.xMin, maxBounds.yMax ) ).transformPosition2( p( maxBounds.xMin, 0 ) ) );
                break;
            }
            
            minBounds = minBounds.withMinX( Math.min( minBounds.xMin, refinedBounds.minBounds.xMin ) );
            maxBounds = maxBounds.withMinX( Math.max( maxBounds.xMin, refinedBounds.maxBounds.xMin ) );
            tempMin = Math.max( tempMin, refinedBounds.maxBounds.yMin );
            tempMax = Math.min( tempMax, refinedBounds.maxBounds.yMax );
        }
        
        // xMax
        tempMin = maxBounds.yMin;
        tempMax = maxBounds.yMax;
        while( isFinite( minBounds.xMax ) && isFinite( maxBounds.xMax ) && Math.abs( minBounds.xMax - maxBounds.xMax ) > precision ) {
            // use maximum bounds except for the x direction, so we don't miss things that we are looking for
            var refinedBounds = scan( idealTransform( new Bounds2( minBounds.xMax, tempMin, maxBounds.xMax, tempMax ) ) );
            
            if( minBounds.xMax >= refinedBounds.minBounds.xMax && maxBounds.xMax <= refinedBounds.maxBounds.xMax ) {
                // sanity check - break out of an infinite loop!
                console.log( 'warning, exiting infinite loop!' );
                break;
            }
            
            minBounds = minBounds.withMaxX( Math.max( minBounds.xMax, refinedBounds.minBounds.xMax ) );
            maxBounds = maxBounds.withMaxX( Math.min( maxBounds.xMax, refinedBounds.maxBounds.xMax ) );
            tempMin = Math.max( tempMin, refinedBounds.maxBounds.yMin );
            tempMax = Math.min( tempMax, refinedBounds.maxBounds.yMax );
        }
        
        // yMin
        tempMin = maxBounds.xMin;
        tempMax = maxBounds.xMax;
        while( isFinite( minBounds.yMin ) && isFinite( maxBounds.yMin ) && Math.abs( minBounds.yMin - maxBounds.yMin ) > precision ) {
            // use maximum bounds except for the y direction, so we don't miss things that we are looking for
            var refinedBounds = scan( idealTransform( new Bounds2( tempMin, maxBounds.yMin, tempMax, minBounds.yMin ) ) );
            
            if( minBounds.yMin <= refinedBounds.minBounds.yMin && maxBounds.yMin >= refinedBounds.maxBounds.yMin ) {
                // sanity check - break out of an infinite loop!
                console.log( 'warning, exiting infinite loop!' );
                break;
            }
            
            minBounds = minBounds.withMinY( Math.min( minBounds.yMin, refinedBounds.minBounds.yMin ) );
            maxBounds = maxBounds.withMinY( Math.max( maxBounds.yMin, refinedBounds.maxBounds.yMin ) );
            tempMin = Math.max( tempMin, refinedBounds.maxBounds.xMin );
            tempMax = Math.min( tempMax, refinedBounds.maxBounds.xMax );
        }
        
        // yMax
        tempMin = maxBounds.xMin;
        tempMax = maxBounds.xMax;
        while( isFinite( minBounds.yMax ) && isFinite( maxBounds.yMax ) && Math.abs( minBounds.yMax - maxBounds.yMax ) > precision ) {
            // use maximum bounds except for the y direction, so we don't miss things that we are looking for
            var refinedBounds = scan( idealTransform( new Bounds2( tempMin, minBounds.yMax, tempMax, maxBounds.yMax ) ) );
            
            if( minBounds.yMax >= refinedBounds.minBounds.yMax && maxBounds.yMax <= refinedBounds.maxBounds.yMax ) {
                // sanity check - break out of an infinite loop!
                console.log( 'warning, exiting infinite loop!' );
                break;
            }
            
            minBounds = minBounds.withMaxY( Math.max( minBounds.yMax, refinedBounds.minBounds.yMax ) );
            maxBounds = maxBounds.withMaxY( Math.min( maxBounds.yMax, refinedBounds.maxBounds.yMax ) );
            tempMin = Math.max( tempMin, refinedBounds.maxBounds.xMin );
            tempMax = Math.min( tempMax, refinedBounds.maxBounds.xMax );
        }
        
        if( debugChromeBoundsScanning ) {
            console.log( 'minBounds: ' + minBounds );
            console.log( 'maxBounds: ' + maxBounds );
        }
        
        var result = new Bounds2(
            ( minBounds.xMin + maxBounds.xMin ) / 2,
            ( minBounds.yMin + maxBounds.yMin ) / 2,
            ( minBounds.xMax + maxBounds.xMax ) / 2,
            ( minBounds.yMax + maxBounds.yMax ) / 2
        );
        
        // extra data about our bounds
        result.minBounds = minBounds;
        result.maxBounds = maxBounds;
        result.isConsistent = maxBounds.containsBounds( minBounds );
        result.precision = Math.max(
            Math.abs( minBounds.xMin - maxBounds.xMin ),
            Math.abs( minBounds.yMin - maxBounds.yMin ),
            Math.abs( minBounds.xMax - maxBounds.xMax ),
            Math.abs( minBounds.yMax - maxBounds.yMax )
        );
        
        // return the average
        return result;
    };
})();