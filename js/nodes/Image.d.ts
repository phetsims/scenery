import Node from './Node';
import {NodeOptions} from './Node';

export type ImageOptions = {
  imageName: string
} & NodeOptions;

export default class Image extends Node {
  constructor( any, options?: Partial<ImageOptions> );
}