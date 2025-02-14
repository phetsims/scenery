// Copyright 2025, University of Colorado Boulder

/**
 * Rasterization utilities for arbitrary Nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Image, { ImageOptions } from '../nodes/Image.js';
import Node, { NodeOptions } from '../nodes/Node.js';
import deprecationWarning from '../../../phet-core/js/deprecationWarning.js';
import optionize, { combineOptions } from '../../../phet-core/js/optionize.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import { roundSymmetric } from '../../../dot/js/util/roundSymmetric.js';

export type RasterizedOptions = {

  // {number} - Controls the resolution of the image relative to the local view units. For example, if our Node is
  // ~100 view units across (in the local coordinate frame) but you want the image to actually have a ~200-pixel
  // resolution, provide resolution:2.
  // Defaults to 1.0
  resolution?: number;

  // {Bounds2|null} - If provided, it will control the x/y/width/height of the toCanvas call. See toCanvas for
  // details on how this controls the rasterization. This is in the "parent" coordinate frame, similar to
  // node.bounds.
  // Defaults to null
  sourceBounds?: Bounds2 | null;

  // {boolean} - If true, the localBounds of the result will be set in a way such that it will precisely match
  // the visible bounds of the original Node (this). Note that antialiased content (with a much lower resolution)
  // may somewhat spill outside these bounds if this is set to true. Usually this is fine and should be the
  // recommended option. If sourceBounds are provided, they will restrict the used bounds (so it will just
  // represent the bounds of the sliced part of the image).
  // Defaults to true
  useTargetBounds?: boolean;

  // {boolean} - If true, the created Image Node gets wrapped in an extra Node so that it can be transformed
  // independently. If there is no need to transform the resulting node, wrap:false can be passed so that no extra
  // Node is created.
  // Defaults to true
  wrap?: boolean;

  // {boolean} - If true, it will directly use the <canvas> element (only works with canvas/webgl renderers)
  // instead of converting this into a form that can be used with any renderer. May have slightly better
  // performance if svg/dom renderers do not need to be used.
  // Defaults to false
  useCanvas?: boolean;

  // Options to be passed to the Node that is returned by the rasterizeNode call, this could be the direct Image or a
  // wrapped Node, depending on the value of options.wrap. In general it is best to use this option, and only provide
  // imageOptions for specific requirements. These options will override any imageOptions if wrap:false. Defaults to \
  // the empty object.
  nodeOptions?: NodeOptions;

  // To be passed to the Image node created from the rasterization. See below for options that will override
  // what is passed in. In general, it is better to use nodeOptions. These options are overridden by nodeOptions when
  // wrap:false. Defaults to the empty object.
  imageOptions?: ImageOptions;
};

// Overload signatures, when wrap is true, the return type is a Node, otherwise it is an Image
export function rasterizeNode( node: Node, providedOptions?: RasterizedOptions & { wrap?: true } ): Node;
export function rasterizeNode( node: Node, providedOptions: RasterizedOptions & { wrap: false } ): Image;

// Implementation signature (must be compatible with the overloads)
/**
 * Returns a Node (backed by a scenery Image) that is a rasterized version of this node. See options, by default the
 * image is wrapped with a container Node.
 */
export function rasterizeNode( node: Node, providedOptions?: RasterizedOptions ): Node | Image {
  const options = optionize<RasterizedOptions, RasterizedOptions>()( {
    resolution: 1,
    sourceBounds: null,
    useTargetBounds: true,
    wrap: true,
    useCanvas: false,
    nodeOptions: {},
    imageOptions: {}
  }, providedOptions );

  const resolution = options.resolution;
  const sourceBounds = options.sourceBounds;

  if ( assert ) {
    assert( typeof resolution === 'number' && resolution > 0, 'resolution should be a positive number' );
    assert( sourceBounds === null || sourceBounds instanceof Bounds2, 'sourceBounds should be null or a Bounds2' );
    if ( sourceBounds ) {
      assert( sourceBounds.isValid(), 'sourceBounds should be valid (finite non-negative)' );
      assert( Number.isInteger( sourceBounds.width ), 'sourceBounds.width should be an integer' );
      assert( Number.isInteger( sourceBounds.height ), 'sourceBounds.height should be an integer' );
    }
  }

  // We'll need to wrap it in a container Node temporarily (while rasterizing) for the scale
  const tempWrapperNode = new Node( {
    scale: resolution,
    children: [ node ]
  } );

  let transformedBounds = sourceBounds || node.getSafeTransformedVisibleBounds().dilated( 2 ).roundedOut();

  // Unfortunately if we provide a resolution AND bounds, we can't use the source bounds directly.
  if ( resolution !== 1 ) {
    transformedBounds = new Bounds2(
      resolution * transformedBounds.minX,
      resolution * transformedBounds.minY,
      resolution * transformedBounds.maxX,
      resolution * transformedBounds.maxY
    );
    // Compensate for non-integral transformedBounds after our resolution transform
    if ( transformedBounds.width % 1 !== 0 ) {
      transformedBounds.maxX += 1 - ( transformedBounds.width % 1 );
    }
    if ( transformedBounds.height % 1 !== 0 ) {
      transformedBounds.maxY += 1 - ( transformedBounds.height % 1 );
    }
  }

  let imageOrNull: Image | null = null;

  // NOTE: This callback is executed SYNCHRONOUSLY
  function callback( canvas: HTMLCanvasElement, x: number, y: number, width: number, height: number ): void {
    const imageSource = options.useCanvas ? canvas : canvas.toDataURL();

    imageOrNull = new Image( imageSource, combineOptions<ImageOptions>( {}, options.imageOptions, {
      x: -x,
      y: -y,
      initialWidth: width,
      initialHeight: height
    } ) );

    // We need to prepend the scale due to order of operations
    imageOrNull.scale( 1 / resolution, 1 / resolution, true );
  }

  // NOTE: Rounding necessary due to floating point arithmetic in the width/height computation of the bounds
  tempWrapperNode.toCanvas( callback, -transformedBounds.minX, -transformedBounds.minY, roundSymmetric( transformedBounds.width ), roundSymmetric( transformedBounds.height ) );

  assert && assert( imageOrNull, 'The toCanvas should have executed synchronously' );
  const image = imageOrNull!;

  tempWrapperNode.dispose();

  // For our useTargetBounds option, we do NOT want to include any "safe" bounds, and instead want to stay true to
  // the original bounds. We do filter out invisible subtrees to set the bounds.
  let finalParentBounds = node.getVisibleBounds();
  if ( sourceBounds ) {
    // If we provide sourceBounds, don't have resulting bounds that go outside.
    finalParentBounds = sourceBounds.intersection( finalParentBounds );
  }

  if ( options.useTargetBounds ) {
    image.imageBounds = image.parentToLocalBounds( finalParentBounds );
  }

  let returnNode: Node;
  if ( options.wrap ) {
    const wrappedNode = new Node( { children: [ image ] } );
    if ( options.useTargetBounds ) {
      wrappedNode.localBounds = finalParentBounds;
    }
    returnNode = wrappedNode;
  }
  else {
    if ( options.useTargetBounds ) {
      image.localBounds = image.parentToLocalBounds( finalParentBounds );
    }
    returnNode = image;
  }

  return returnNode.mutate( options.nodeOptions );
}

/**
 * Calls the callback with an Image Node that contains this Node's subtree's visual form. This is always
 * asynchronous, but the resulting image Node can be used with any back-end (Canvas/WebGL/SVG/etc.)
 * @deprecated - Use rasterizeNode() instead (should avoid the asynchronous-ness)
 *
 * @param callback - callback( imageNode {Image} ) is called
 * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
 * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
 * @param [width] - The width of the Canvas output
 * @param [height] - The height of the Canvas output
 */
export const toImageNodeAsynchronous = ( node: Node, callback: ( image: Node ) => void, x?: number, y?: number, width?: number, height?: number ): void => {

  assert && deprecationWarning( 'Node.toImageNodeAsyncrhonous() is deprecated, please use rasterizeNode() instead' );

  assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
  assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
  assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
    'If provided, width should be a non-negative integer' );
  assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
    'If provided, height should be a non-negative integer' );

  node.toImage( ( image, x, y ) => {
    callback( new Node( {
      children: [
        new Image( image, { x: -x, y: -y } )
      ]
    } ) );
  }, x, y, width, height );
};

/**
 * Creates a Node containing an Image Node that contains this Node's subtree's visual form. This is always
 * synchronous, but the resulting image Node can ONLY used with Canvas/WebGL (NOT SVG).
 * @deprecated - Use rasterizeNode() instead, should be mostly equivalent if useCanvas:true is provided.
 *
 * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
 * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
 * @param [width] - The width of the Canvas output
 * @param [height] - The height of the Canvas output
 */
export const toCanvasNodeSynchronous = ( node: Node, x?: number, y?: number, width?: number, height?: number ): Node => {

  assert && deprecationWarning( 'Node.toCanvasNodeSynchronous() is deprecated, please use rasterizeNode() instead' );

  assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
  assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
  assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
    'If provided, width should be a non-negative integer' );
  assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
    'If provided, height should be a non-negative integer' );

  let result: Node | null = null;
  node.toCanvas( ( canvas, x, y ) => {
    result = new Node( {
      children: [
        new Image( canvas, { x: -x, y: -y } )
      ]
    } );
  }, x, y, width, height );
  assert && assert( result, 'toCanvasNodeSynchronous requires that the node can be rendered only using Canvas' );
  return result!;
};

/**
 * Returns an Image that renders this Node. This is always synchronous, and sets initialWidth/initialHeight so that
 * we have the bounds immediately.  Use this method if you need to reduce the number of parent Nodes.
 *
 * NOTE: the resultant Image should be positioned using its bounds rather than (x,y).  To create a Node that can be
 * positioned like any other node, please use toDataURLNodeSynchronous.
 * @deprecated - Use rasterizeNode() instead, should be mostly equivalent if wrap:false is provided.
 *
 * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
 * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
 * @param [width] - The width of the Canvas output
 * @param [height] - The height of the Canvas output
 */
export const toDataURLImageSynchronous = ( node: Node, x?: number, y?: number, width?: number, height?: number ): Image => {

  assert && deprecationWarning( 'Node.toDataURLImageSychronous() is deprecated, please use rasterizeNode() instead' );

  assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
  assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
  assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
    'If provided, width should be a non-negative integer' );
  assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
    'If provided, height should be a non-negative integer' );

  let result: Image | null = null;
  node.toDataURL( ( dataURL, x, y, width, height ) => {
    result = new Image( dataURL, { x: -x, y: -y, initialWidth: width, initialHeight: height } );
  }, x, y, width, height );
  assert && assert( result, 'toDataURL failed to return a result synchronously' );
  return result!;
};

/**
 * Returns a Node that contains this Node's subtree's visual form. This is always synchronous, and sets
 * initialWidth/initialHeight so that we have the bounds immediately.  An extra wrapper Node is provided
 * so that transforms can be done independently.  Use this method if you need to be able to transform the node
 * the same way as if it had not been rasterized.
 * @deprecated - Use rasterizeNode() instead, should be mostly equivalent
 *
 * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
 * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
 * @param [width] - The width of the Canvas output
 * @param [height] - The height of the Canvas output
 */
export const toDataURLNodeSynchronous = ( node: Node, x?: number, y?: number, width?: number, height?: number ): Node => {

  assert && deprecationWarning( 'Node.toDataURLNodeSynchronous() is deprecated, please use rasterizeNode() instead' );

  assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
  assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
  assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
    'If provided, width should be a non-negative integer' );
  assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
    'If provided, height should be a non-negative integer' );

  return new Node( {
    children: [
      toDataURLImageSynchronous( node, x, y, width, height )
    ]
  } );
};