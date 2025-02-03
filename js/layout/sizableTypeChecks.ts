// Copyright 2025, University of Colorado Boulder

/**
 * To avoid circular dependencies, this file contains type checks for the Sizable type hierarchy.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import type { TWidthSizable } from './WidthSizable.js';
import type Node from '../nodes/Node.js';
import type { THeightSizable } from './HeightSizable.js';
import type { TSizable } from './Sizable.js';

// Some typescript gymnastics to provide a user-defined type guard that treats something as widthSizable
// We need to define an unused function with a concrete type, so that we can extract the return type of the function
// and provide a type for a Node that extends this type.
export type WidthSizableNode = Node & TWidthSizable;
export type HeightSizableNode = Node & THeightSizable;
export type SizableNode = Node & TSizable;

export const isWidthSizable = ( node: Node ): node is WidthSizableNode => {
  return node.widthSizable;
};
export const isHeightSizable = ( node: Node ): node is HeightSizableNode => {
  return node.heightSizable;
};
export const isSizable = ( node: Node ): node is SizableNode => {
  return node.widthSizable && node.heightSizable;
};

export const extendsWidthSizable = ( node: Node ): node is WidthSizableNode => {
  return node.extendsWidthSizable;
};
export const extendsHeightSizable = ( node: Node ): node is HeightSizableNode => {
  return node.extendsHeightSizable;
};
export const extendsSizable = ( node: Node ): node is SizableNode => {
  return node.extendsSizable;
};