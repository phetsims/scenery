// Copyright 2022, University of Colorado Boulder

/**
 * Deserializes a generalized object
 * @deprecated
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';
import { Circle, Color, DOM, Gradient, Image, Line, LinearGradient, Mipmap, Node, Paint, Path, Pattern, RadialGradient, Rectangle, scenery, Text } from '../imports.js';

const sceneryDeserialize = ( value: { type: string; [ key: string ]: IntentionalAny } ): IntentionalAny => {
  const nodeTypes = [
    'Node', 'Path', 'Circle', 'Line', 'Rectangle', 'Text', 'Image', 'CanvasNode', 'WebGLNode', 'DOM'
  ];

  if ( value.type === 'Vector2' ) {
    return new Vector2( value.x, value.y );
  }
  if ( value.type === 'Matrix3' ) {
    return new Matrix3().rowMajor(
      value.m00, value.m01, value.m02,
      value.m10, value.m11, value.m12,
      value.m20, value.m21, value.m22
    );
  }
  else if ( value.type === 'Bounds2' ) {
    return new Bounds2( value.minX, value.minY, value.maxX, value.maxY );
  }
  else if ( value.type === 'Shape' ) {
    return new Shape( value.path );
  }
  else if ( value.type === 'Array' ) {
    return value.value.map( sceneryDeserialize );
  }
  else if ( value.type === 'Color' ) {
    return new Color( value.red, value.green, value.blue, value.alpha );
  }
  else if ( value.type === 'Property' ) {
    return new Property( sceneryDeserialize( value.value ) );
  }
  else if ( value.type === 'Pattern' || value.type === 'LinearGradient' || value.type === 'RadialGradient' ) {
    let paint!: Paint;

    if ( value.type === 'Pattern' ) {
      const img = new window.Image();
      img.src = value.url;
      paint = new Pattern( img );
    }
    // Gradients
    else {
      const start = sceneryDeserialize( value.start );
      const end = sceneryDeserialize( value.end );
      if ( value.type === 'LinearGradient' ) {
        paint = new LinearGradient( start.x, start.y, end.x, end.y );
      }
      else if ( value.type === 'RadialGradient' ) {
        paint = new RadialGradient( start.x, start.y, value.startRadius, end.x, end.y, value.endRadius );
      }

      value.stops.forEach( ( stop: IntentionalAny ) => {
        ( paint as Gradient ).addColorStop( stop.ratio, sceneryDeserialize( stop.stop ) );
      } );
    }

    if ( value.transformMatrix ) {
      paint.setTransformMatrix( sceneryDeserialize( value.transformMatrix ) );
    }

    return paint;
  }
  else if ( _.includes( nodeTypes, value.type ) ) {
    let node: IntentionalAny;

    const setup = value.setup;

    if ( value.type === 'Node' ) {
      node = new Node();
    }
    else if ( value.type === 'Path' ) {
      node = new Path( sceneryDeserialize( setup.path ) );
    }
    else if ( value.type === 'Circle' ) {
      node = new Circle( {} );
    }
    else if ( value.type === 'Line' ) {
      node = new Line( {} );
    }
    else if ( value.type === 'Rectangle' ) {
      node = new Rectangle( {} );
    }
    else if ( value.type === 'Text' ) {
      node = new Text( '' );
    }
    else if ( value.type === 'Image' ) {
      if ( setup.imageType === 'image' || setup.imageType === 'canvas' ) {
        node = new Image( setup.src );
        if ( setup.generateMipmaps ) {
          node.mipmap = true;
        }
      }
      else if ( setup.imageType === 'mipmapData' ) {
        const mipmapData = setup.mipmapData.map( ( level: Mipmap[0] ) => {
          const result: IntentionalAny = {
            width: level.width,
            height: level.height,
            url: level.url
          };
          result.img = new window.Image();
          result.img.src = level.url;
          result.canvas = document.createElement( 'canvas' );
          result.canvas.width = level.width;
          result.canvas.height = level.height;
          const context = result.canvas.getContext( '2d' );
          // delayed loading like in mipmap plugin
          result.updateCanvas = function() {
            if ( result.img.complete && ( typeof result.img.naturalWidth === 'undefined' || result.img.naturalWidth > 0 ) ) {
              context.drawImage( result.img, 0, 0 );
              delete result.updateCanvas;
            }
          };
          return result;
        } );
        node = new Image( mipmapData );
      }
      ( node! ).initialWidth = setup.width;
      ( node! ).initialHeight = setup.height;
    }
    else if ( value.type === 'CanvasNode' || value.type === 'WebGLNode' ) {
      // TODO: Record Canvas/WebGL calls? (conditionals would be harder!)
      node = new Node( {
        children: [
          new Image( setup.url, {
            translation: sceneryDeserialize( setup.offset ),
            scale: 1 / setup.scale
          } )
        ]
      } );
    }
    else if ( value.type === 'DOM' ) {
      const div = document.createElement( 'div' );
      div.innerHTML = value.element;
      const element = div.childNodes[ 0 ];
      div.removeChild( element );

      if ( value.dataURL ) {
        const img = new window.Image();
        img.onload = () => {
          const context = ( element as HTMLCanvasElement ).getContext( '2d' )!;
          context.drawImage( img, 0, 0 );
        };
        img.src = value.dataURL;
      }

      node = new DOM( element as HTMLElement );
    }

    if ( setup.clipArea ) {
      ( node! ).clipArea = sceneryDeserialize( setup.clipArea );
    }
    if ( setup.mouseArea ) {
      ( node! ).mouseArea = sceneryDeserialize( setup.mouseArea );
    }
    if ( setup.touchArea ) {
      ( node! ).touchArea = sceneryDeserialize( setup.touchArea );
    }
    if ( setup.matrix ) {
      ( node! ).matrix = sceneryDeserialize( setup.matrix );
    }
    if ( setup.localBounds ) {
      ( node! ).localBounds = sceneryDeserialize( setup.localBounds );
    }

    // Paintable, if they exist
    if ( setup.fill ) {
      ( node as Path | Text ).fill = sceneryDeserialize( setup.fill );
    }
    if ( setup.stroke ) {
      ( node as Path | Text ).stroke = sceneryDeserialize( setup.stroke );
    }
    if ( setup.lineDash ) {
      ( node as Path | Text ).lineDash = sceneryDeserialize( setup.lineDash );
    }

    ( node! ).mutate( value.options );

    ( node! )._serialization = value;

    return node;
  }
  else if ( value.type === 'Subtree' ) {
    const nodeMap: Record<string, Node> = {};
    const nodes = value.nodes.map( sceneryDeserialize );

    // Index them
    nodes.forEach( ( node: Node ) => {
      nodeMap[ node._serialization.id ] = node;
    } );

    // Connect children
    nodes.forEach( ( node: Node ) => {
      node._serialization.setup.children.forEach( ( childId: string ) => {
        node.addChild( nodeMap[ childId ] );
      } );
    } );

    // The root should be the first one
    return nodeMap[ value.rootNodeId ];
  }
  else if ( value.type === 'value' ) {
    return value.value;
  }
  else {
    return null;
  }
};

scenery.register( 'sceneryDeserialize', sceneryDeserialize );
export default sceneryDeserialize;
